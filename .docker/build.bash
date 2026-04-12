#!/usr/bin/env bash

# macOS-compatible readlink -f alternative
realpath_portable() {
    local path="$1"
    if [[ "$(uname)" == "Darwin" ]]; then
        python3 -c "import os; print(os.path.realpath('$path'))"
    else
        readlink -f "$path"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "$(realpath_portable "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Fixed image tag
TAG="panda_gz_moveit2"

if [ "${#}" -gt "0" ]; then
    if [[ "${1}" != "-"* ]]; then
        TAG="${TAG}:${1}"
        BUILD_ARGS=${*:2}
    else
        BUILD_ARGS=${*:1}
    fi
fi

# Detect platform for native builds (avoids slow emulation on Apple Silicon)
if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    PLATFORM_FLAG="--platform linux/arm64"
elif [[ "$(uname -m)" == "x86_64" ]]; then
    PLATFORM_FLAG="--platform linux/amd64"
else
    PLATFORM_FLAG=""
fi

DOCKER_BUILD_CMD=(
    docker build
    ${PLATFORM_FLAG}
    "${PROJECT_DIR}"
    --tag "${TAG}"
    "${BUILD_ARGS}"
)

echo -e "\033[1;30m${DOCKER_BUILD_CMD[*]}\033[0m" | xargs

# shellcheck disable=SC2048
exec ${DOCKER_BUILD_CMD[*]}
