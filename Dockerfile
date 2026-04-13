ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Define working directory
ARG WS_DIR=/root/ws
ENV WS_DIR=${WS_DIR}
ENV WS_SRC_DIR=${WS_DIR}/src
ENV WS_INSTALL_DIR=${WS_DIR}/install
ENV WS_LOG_DIR=${WS_DIR}/log
WORKDIR ${WS_DIR}

### Install Gazebo, graphics libraries, GPD, and VNC for remote display
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -yq --no-install-recommends \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-py-trees-ros-viewer \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libvulkan1 \
    mesa-vulkan-drivers \
    tmux \
    libpcl-dev \
    libopencv-dev \
    libeigen3-dev \
    tigervnc-standalone-server \
    tigervnc-tools \
    openbox \
    xterm \
    python3-pip \
    alsa-utils \
    pulseaudio-utils \
    libasound2-plugins \
    libportaudio2 \
    espeak \
    python3-pyaudio && \
    rm -rf /var/lib/apt/lists/*

### Install Python audio libraries and pre-download Vosk model
RUN pip3 install --break-system-packages vosk sounddevice pyttsx3 && \
    python3 -c "from vosk import Model; Model(lang='en-us')"

### Route ALSA to PulseAudio
RUN echo -e "pcm.!default {\n    type pulse\n}\nctl.!default {\n    type pulse\n}" > /etc/asound.conf

### NVIDIA environment variables for graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

### Import and install dependencies, then build these dependencies (not panda_gz_moveit2 yet)
COPY ./panda_gz_moveit2.repos ${WS_SRC_DIR}/panda_gz_moveit2/panda_gz_moveit2.repos
RUN vcs import --shallow ${WS_SRC_DIR} < ${WS_SRC_DIR}/panda_gz_moveit2/panda_gz_moveit2.repos && \
    rosdep update && \
    apt-get update && \
    rosdep install -y -r -i --rosdistro "${ROS_DISTRO}" --from-paths ${WS_SRC_DIR} && \
    rm -rf /var/lib/apt/lists/* && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release" && \
    rm -rf ${WS_LOG_DIR}

### Copy over the rest of panda_gz_moveit2, then install dependencies and build
COPY ./ ${WS_SRC_DIR}/panda_gz_moveit2/
RUN rosdep update && \
    apt-get update && \
    rosdep install -y -r -i --rosdistro "${ROS_DISTRO}" --from-paths ${WS_SRC_DIR} && \
    rm -rf /var/lib/apt/lists/* && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release" && \
    rm -rf ${WS_LOG_DIR}

### Pre-configure GPD so students can build with one command
RUN cmake -S ${WS_SRC_DIR}/panda_gz_moveit2/deps/gpd \
    -B ${WS_SRC_DIR}/panda_gz_moveit2/deps/gpd/build \
    -DCMAKE_BUILD_TYPE=Release

### Add workspace to the ROS entrypoint
### Source ROS workspace inside `~/.bashrc` to enable autocompletion
### Add convenience aliases
RUN sed -i '$i source "${WS_INSTALL_DIR}/local_setup.bash" --' /ros_entrypoint.sh && \
    sed -i '$a source "/opt/ros/${ROS_DISTRO}/setup.bash"' ~/.bashrc && \
    echo 'alias launch_ctrl="ros2 launch panda_moveit_config ex_gz_control.launch.py"' >> ~/.bashrc && \
    echo 'alias launch_bt="ros2 run panda_moveit_config bt_pick_place.py"' >> ~/.bashrc && \
    echo 'alias build="colcon build --merge-install --symlink-install --cmake-args \"-DCMAKE_BUILD_TYPE=Release\""' >> ~/.bashrc
