#!/usr/bin/env bash

# Make Docker Session folder
if [ ! -d "session" ]; then
	mkdir session
fi

# Make Docker Volume
sudo docker volume create --name session
sudo cp -a session/. /var/lib/docker/volumes/session/_data

# Run DeepDoom-DE Docker using GPU
zero=0;
if [ $# -eq 0 ] 
	then
   		sudo nvidia-docker run  -ti --privileged --net=host \
      		-v /tmp/.X11-unix:/tmp/.X11-unix \
     		-v session:/home/DeepDoom-DE/session \
      		-e DISPLAY=${DISPLAY} \
      		--rm deepdoomde-gpu
	else
		sudo nvidia-docker run  -ti --privileged --net=host \
      		-v /tmp/.X11-unix:/tmp/.X11-unix \
     		-v session:/home/DeepDoom-DE/session \
      		-e DISPLAY=${DISPLAY} \
      		--rm deepdoomde-gpu $1
fi

# Copy Volume Data to Docker Session Folder
sudo cp -rfp -a /var/lib/docker/volumes/session/_data/. session
sudo chown -R $USER: session
sudo chmod -R u+w session
sudo docker volume rm session