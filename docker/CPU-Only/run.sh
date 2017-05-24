#!/usr/bin/env bash

# Run DeepDoom-DE Docker using CPU
sudo docker run  -ti --privileged --net=host \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -e DISPLAY=${DISPLAY} \
      --rm deepdoomde-cpu
