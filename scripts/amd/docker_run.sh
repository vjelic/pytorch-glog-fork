set -o xtrace

alias drun='sudo docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

# DEVICES="--gpus all"
DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR="/root/$(basename $(pwd))"
WORK_DIR="/dockerx/$(basename $(pwd))"

# IMAGE_NAME=nvcr.io/nvidia/pytorch:21.08-py3
# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:8733_ubuntu18.04_py3.6_pytorch_rocm4.5_internal_testing_warpsize_a1f2f40
# IMAGE_NAME=computecqe/pytorch:8863_warpsize32
# IMAGE_NAME=rocm/pytorch-private:SWDEV-navi
# IMAGE_NAME=computecqe/pytorch:rocm4.5_internal_testing_warpsize_mmelesse_pr_2
# IMAGE_NAME=rocm/pytorch-private:SWDEV-fixnosymbol
# IMAGE_NAME=rocm/pytorch-private:SWDEV-navihang
# IMAGE_NAME=rocm/pytorch-private:SWDEV-hangsols 
IMAGE_NAME=rocm/pytorch-private:SWDEV-printsleepkernel
CONTAINER_NAME=pytorch

# start new container
CONTAINER_ID=$(drun -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
