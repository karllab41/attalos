build: depends
	docker build -t l41-nvidia-base -f Dockerfile.nvidia .
	docker build -t l41-theano-base -f Dockerfile.theano .
	docker build -t l41-keras-base -f Dockerfile.keras .
	docker build -t l41-caffe-keras-tf -f Dockerfile.caffe-keras-tf .

attalos-bash: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0  -it l41-caffe-keras-tf /bin/bash

notebook: depends
	# docker build -t l41-attalos-notebook -f Dockerfile.notebook .
	docker build -t l41-attalos-notebook-conda -f Dockerfile.notebook .
	docker run -d -p 9999:8888 -v /data:/data -v ~/:/work --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0 -it l41-attalos-notebook /bootstrap.sh

depends:
	@echo
	@echo "checking dependencies"
	@echo
	docker -v
