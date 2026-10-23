FROM osrf/ros:humble-desktop-full@sha256:1db1e4e941d4f77fab55bcd479158a75273329335e86ec468bff280859e8178c

ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=0 \
    GALLIUM_DRIVER=zink \
    MESA_LOADER_DRIVER_OVERRIDE=zink \
    ZINK_DESCRIPTORS=lazy \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/virtio_icd.x86_64.json \
    QT_X11_NO_MITSHM=1 \
    ROS_LOCALHOST_ONLY=1 \
    ROS_DOMAIN_ID=47

# Public PPA signing key; verify its identity before enabling the repository.
COPY keys/kisak-turtle.asc /usr/share/keyrings/kisak-turtle.asc
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && gpg --batch --show-keys --with-colons /usr/share/keyrings/kisak-turtle.asc > /tmp/turtle-keyinfo \
    && awk -F: '$1 == "pub" { pubs++ } $1 == "fpr" && !fingerprint { fingerprint=$10 } END { exit !(pubs == 1 && fingerprint == "EB8B81E14DA65431D7504EA8F63F0F2B90935439") }' /tmp/turtle-keyinfo \
    && printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/kisak-turtle.asc] https://ppa.launchpadcontent.net/kisak/turtle/ubuntu jammy main' > /etc/apt/sources.list.d/kisak-turtle.list \
    && rm -f /tmp/turtle-keyinfo \
    && rm -rf /var/lib/apt/lists/*

# Official X2 runtime/build dependencies plus the pinned Venus/zink GPU stack.
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
      build-essential cmake libcurl4-openssl-dev \
      libeigen3-dev libgflags-dev libgl1 libncurses-dev liboctomap1.9 \
      libopencv-dev libscrypt0 libyaml-cpp-dev mesa-utils pkg-config \
      python3-colcon-common-extensions python3-pip \
      ros-humble-cv-bridge ros-humble-grid-map-msgs ros-humble-grid-map-ros \
      ros-humble-joy ros-humble-rmw-fastrtps-cpp \
      xvfb fluxbox x11vnc novnc websockify x11-utils xfonts-base xterm \
      libgl1-mesa-dri=25.0.7~kisak4~j \
      libglx-mesa0=25.0.7~kisak4~j \
      libegl-mesa0=25.0.7~kisak4~j \
      libgbm1=25.0.7~kisak4~j \
      libgbm-dev=25.0.7~kisak4~j \
      mesa-libgallium=25.0.7~kisak4~j \
      mesa-vulkan-drivers=25.0.7~kisak4~j \
      libdrm-common=2.4.122-1~kisak1~j \
      libdrm-dev=2.4.122-1~kisak1~j \
      libdrm2=2.4.122-1~kisak1~j \
      libdrm-intel1=2.4.122-1~kisak1~j \
      libdrm-radeon1=2.4.122-1~kisak1~j \
      libdrm-nouveau2=2.4.122-1~kisak1~j \
      libdrm-amdgpu1=2.4.122-1~kisak1~j \
    && curl --fail --location --retry 3 \
      https://github.com/VirtualGL/virtualgl/releases/download/3.1.5/virtualgl_3.1.5_amd64.deb \
      --output /tmp/virtualgl_3.1.5_amd64.deb \
    && printf '%s\n' 'df3f7788ce41b182a47c0d298e5cd6d2d63579522cb41825970b7726e825485e  /tmp/virtualgl_3.1.5_amd64.deb' | sha256sum --check - \
    && apt-get install -y --no-install-recommends /tmp/virtualgl_3.1.5_amd64.deb \
    && rm -f /tmp/virtualgl_3.1.5_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/shell-env.sh /etc/profile.d/x2.sh
COPY scripts/entrypoint.sh scripts/desktop.sh /usr/local/bin/
COPY scripts/start-component.sh /usr/local/bin/x2-component
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/desktop.sh /usr/local/bin/x2-component \
    && printf '\nsource /etc/profile.d/x2.sh\n' >> /root/.bashrc \
    && mkdir -p /workspace /agibot/data/var

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# The desktop does not start Sim, MC or a brain process.
CMD ["/usr/local/bin/desktop.sh"]
