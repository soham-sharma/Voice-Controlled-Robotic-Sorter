#!/usr/bin/env bash

# macOS-compatible readlink -f alternative
realpath_portable() {
    local path="$1"
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS: use python as fallback since readlink -f doesn't exist
        python3 -c "import os; print(os.path.realpath('$path'))"
    else
        readlink -f "$path"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "$(realpath_portable "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Fixed image tag
TAG="panda_gz_moveit2"

## Forward custom volumes and environment variables
CUSTOM_VOLUMES=()
CUSTOM_ENVS=()
GPU_OPT=()
GPU_ENVS=()
while getopts ":v:e:" opt; do
    case "${opt}" in
        v) CUSTOM_VOLUMES+=("${OPTARG}") ;;
        e) CUSTOM_ENVS+=("${OPTARG}") ;;
        *)
            echo >&2 "Usage: ${0} [-v VOLUME] [-e ENV] [TAG] [CMD]"
            exit 2
            ;;
    esac
done
shift "$((OPTIND - 1))"

## Determine TAG and CMD positional arguments
if [ "${#}" -gt "0" ]; then
    if [[ $(docker images --format "{{.Tag}}" "${TAG}") =~ (^|[[:space:]])${1}($|[[:space:]]) || $(wget -q https://registry.hub.docker.com/v2/repositories/${TAG}/tags -O - | grep -Poe '(?<=(\"name\":\")).*?(?=\")') =~ (^|[[:space:]])${1}($|[[:space:]]) ]]; then
        # Use the first argument as a tag is such tag exists either locally or on the remote registry
        TAG="${TAG}:${1}"
        CMD=${*:2}
    else
        CMD=${*:1}
    fi
fi

## GPU
# Check for NVIDIA GPU and verify container toolkit is available (Linux only)
if [[ "$(uname)" == "Linux" ]]; then
    LS_HW_DISPLAY=$(lshw -C display 2>/dev/null | grep vendor)
fi
if [[ "$(echo "${LS_HW_DISPLAY:-}" | tr '[:lower:]' '[:upper:]')" =~ NVIDIA ]]; then
    # Test if NVIDIA container toolkit is working
    if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &>/dev/null; then
        if dpkg --compare-versions "$(docker version --format '{{.Server.Version}}')" gt "19.3"; then
            GPU_OPT=("--gpus" "all")
        else
            GPU_OPT=("--runtime" "nvidia")
        fi
        GPU_ENVS=(
            NVIDIA_VISIBLE_DEVICES="all"
            NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics"
        )
        echo -e "\033[0;32mNVIDIA GPU support enabled\033[0m"
    else
        echo -e "\033[0;33mWarning: NVIDIA GPU detected but container toolkit not available. Running without GPU acceleration.\033[0m"
        echo -e "\033[0;33mInstall nvidia-container-toolkit for GPU support.\033[0m"
    fi
fi

## GUI
# To enable GUI, make sure processes in the container can connect to the x server
XAUTH=/tmp/.docker.xauth
if [ ! -f ${XAUTH} ]; then
    touch ${XAUTH}
    chmod a+r ${XAUTH}

    XAUTH_LIST=$(xauth nlist "${DISPLAY}")
    if [ -n "${XAUTH_LIST}" ]; then
        # shellcheck disable=SC2001
        XAUTH_LIST=$(sed -e 's/^..../ffff/' <<<"${XAUTH_LIST}")
        echo "${XAUTH_LIST}" | xauth -f ${XAUTH} nmerge -
    fi
fi
# GUI-enabling volumes
GUI_VOLUMES=(
    "${XAUTH}:${XAUTH}"
)
# X11 socket and input devices (Linux only)
if [[ "$(uname)" == "Linux" ]]; then
    GUI_VOLUMES+=("/tmp/.X11-unix:/tmp/.X11-unix")
    GUI_VOLUMES+=("/dev/input:/dev/input")
elif [[ -d "/tmp/.X11-unix" ]]; then
    # macOS with XQuartz may have this
    GUI_VOLUMES+=("/tmp/.X11-unix:/tmp/.X11-unix")
fi
# GUI-enabling environment variables
GUI_ENVS=(
    XAUTHORITY="${XAUTH}"
    QT_X11_NO_MITSHM=1
    DISPLAY="${DISPLAY}"
)

## Additional volumes
# Synchronize timezone with host (Linux only - macOS handles this differently)
if [[ "$(uname)" == "Linux" && -f "/etc/localtime" ]]; then
    CUSTOM_VOLUMES+=("/etc/localtime:/etc/localtime:ro")
fi

## Additional environment variables
# Synchronize ROS_DOMAIN_ID with host
if [ -n "${ROS_DOMAIN_ID}" ]; then
    CUSTOM_ENVS+=("ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
# Synchronize GZ_PARTITION with host (also check legacy IGN_PARTITION)
if [ -n "${GZ_PARTITION}" ]; then
    CUSTOM_ENVS+=("GZ_PARTITION=${GZ_PARTITION}")
elif [ -n "${IGN_PARTITION}" ]; then
    CUSTOM_ENVS+=("GZ_PARTITION=${IGN_PARTITION}")
fi
# Synchronize RMW configuration with host
if [ -n "${RMW_IMPLEMENTATION}" ]; then
    CUSTOM_ENVS+=("RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
fi
if [ -n "${CYCLONEDDS_URI}" ]; then
    CUSTOM_ENVS+=("CYCLONEDDS_URI=${CYCLONEDDS_URI}")
    CUSTOM_VOLUMES+=("${CYCLONEDDS_URI//file:\/\//}:${CYCLONEDDS_URI//file:\/\//}:ro")
fi
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]; then
    CUSTOM_ENVS+=("FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}")
    CUSTOM_VOLUMES+=("${FASTRTPS_DEFAULT_PROFILES_FILE}:${FASTRTPS_DEFAULT_PROFILES_FILE}:ro")
fi

DOCKER_RUN_CMD=(
    docker run
    --interactive
    --tty
    --rm
    --network host
    --ipc host
    --privileged
    --security-opt "seccomp=unconfined"
    -v "$(pwd):/root/ws/src/panda_gz_moveit2:rw"
)
# Add GUI volumes
for vol in "${GUI_VOLUMES[@]}"; do
    DOCKER_RUN_CMD+=("--volume" "${vol}")
done
# Add GUI environment variables
for env in "${GUI_ENVS[@]}"; do
    DOCKER_RUN_CMD+=("--env" "${env}")
done
# Add GPU options only if set
if [ ${#GPU_OPT[@]} -gt 0 ]; then
    DOCKER_RUN_CMD+=("${GPU_OPT[@]}")
fi
for env in "${GPU_ENVS[@]:-}"; do
    [ -n "${env}" ] && DOCKER_RUN_CMD+=("--env" "${env}")
done
for vol in "${CUSTOM_VOLUMES[@]:-}"; do
    [ -n "${vol}" ] && DOCKER_RUN_CMD+=("--volume" "${vol}")
done
for env in "${CUSTOM_ENVS[@]:-}"; do
    [ -n "${env}" ] && DOCKER_RUN_CMD+=("--env" "${env}")
done
DOCKER_RUN_CMD+=("${TAG}")
[ -n "${CMD:-}" ] && DOCKER_RUN_CMD+=("${CMD}")

echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m"

exec "${DOCKER_RUN_CMD[@]}"
