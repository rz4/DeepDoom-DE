bootstrap:docker
From:ubuntu:16.04
nvidia/cuda:8.0-cudnn5-devel

%post
apt-get update
apt-get install -y \

    build-essential \

    bzip2 \

    cmake \

    curl \

    git \

    libboost-all-dev \

    libbz2-dev \

    libfluidsynth-dev \

    libfreetype6-dev \

    libgme-dev \

    libgtk2.0-dev \

    libjpeg-dev \

    libopenal-dev \

    libpng12-dev \

    libsdl2-dev \

    libwildmidi-dev \

    libzmq3-dev \

    nano \

    nasm \

    pkg-config \

    rsync \

    software-properties-common \

    sudo \

    tar \

    timidity \

    unzip \

    wget \

    zlib1g-dev \

    pulseaudio
apt-get install -y python3-dev python3 python3-pip python3-tk
pip3 install pip --upgrade
pip3 install numpy
export USER_NAME=DeepDoom-DE
echo "
export USER_NAME=DeepDoom-DE" >> /environment
export HOME_DIR=/home/$USER_NAME
echo "
export HOME_DIR=/home/$USER_NAME" >> /environment
export HOST_UID=1000
echo "
export HOST_UID=1000" >> /environment
export HOST_GID=1000
echo "
export HOST_GID=1000" >> /environment
export uid=${HOST_UID} gid=${HOST_GID} && \

    mkdir -p ${HOME_DIR} && \

    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \

    echo "$USER_NAME:x:${uid}:" >> /etc/group && \

    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \

    chmod 0440 /etc/sudoers.d/$USER_NAME && \

    chown ${uid}:${gid} -R ${HOME_DIR}
pip3 install git+https://github.com/rz4/DeepDoom-DE
pip3 install tensorflow-gpu

USER ${USER_NAME}
cd ${HOME_DIR}
echo 'PS1="\[$(tput bold)\]\[\033[38;5;10m\]\u@Docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\[\033[38;5;15m\]\\$\[$(tput sgr0)\]"' >> .bashrc
/bin/bashmkdir session
cd ${HOME_DIR}/session
pax11publish -r; pulseaudio --start
