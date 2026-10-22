FROM osrf/ros:humble-desktop-full@sha256:1db1e4e941d4f77fab55bcd479158a75273329335e86ec468bff280859e8178c

ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    LP_NUM_THREADS=2 \
    QT_X11_NO_MITSHM=1 \
    ROS_LOCALHOST_ONLY=1 \
    ROS_DOMAIN_ID=47

# Runtime/build dependencies from the official X2 sim_contents.html Dockerfile.
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl libcurl4-openssl-dev \
      libeigen3-dev libgflags-dev libgl1 libgl1-mesa-dri libglx-mesa0 \
      libncurses-dev liboctomap1.9 libopencv-dev libscrypt0 libyaml-cpp-dev \
      mesa-utils pkg-config python3-colcon-common-extensions python3-pip \
      ros-humble-cv-bridge ros-humble-grid-map-msgs ros-humble-grid-map-ros \
      ros-humble-joy ros-humble-rmw-fastrtps-cpp \
      xvfb fluxbox x11vnc novnc websockify x11-utils xfonts-base xterm \
    && rm -rf /var/lib/apt/lists/*

COPY shell-env.sh /etc/profile.d/x2.sh
COPY entrypoint.sh desktop.sh /usr/local/bin/
COPY start-component.sh /usr/local/bin/x2-component
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/desktop.sh /usr/local/bin/x2-component \
    && printf '\nsource /etc/profile.d/x2.sh\n' >> /root/.bashrc \
    && mkdir -p /workspace /agibot/data/var

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/local/bin/desktop.sh"]
