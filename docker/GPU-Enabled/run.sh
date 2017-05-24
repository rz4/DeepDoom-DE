#!/usr/bin/env bash

# Run DeepDoom-DE Docker using GPU
sudo nvidia-docker run  -ti --privileged --net=host \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /dev/shm:/dev/shm \
      -v /etc/machine-id:/etc/machine-id \
      -v /run/user/$uid/pulse:/run/user/$uid/pulse \
      -v /var/lib/dbus:/var/lib/dbus \
      -v ~/.pulse:/home/$dockerUsername/.pulse \
      -e DISPLAY=${DISPLAY} \
      --rm deepdoomde-gpu 
