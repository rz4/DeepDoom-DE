FROM ubuntu:16.04

# ViZdoom dependencies
RUN apt-get update
RUN apt-get install -y \
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

# Python3
RUN apt-get install -y python3-dev python3 python3-pip python3-tk
RUN pip3 install pip --upgrade
RUN pip3 install numpy

# Enables X11 sharing and creates user home directory
ENV USER_NAME DeepDoom-DE
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

RUN pip3 install git+https://github.com/rz4/DeepDoom-DE

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

RUN echo 'PS1="\[$(tput bold)\]\[\033[38;5;10m\]\u@Docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\[\033[38;5;15m\]\\$\[$(tput sgr0)\]"' >> .bashrc
CMD ["/bin/bash"]

CMD pax11publish -r; pulseaudio --start
CMD python3 -m deepdoomde; 'bash'
