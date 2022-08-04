set -o xtrace

alias drun='sudo docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

# DEVICES="--gpus all"
DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR="/root/$(basename $(pwd))"
# WORK_DIR="/dockerx/$(basename $(pwd))"
WORK_DIR="/var/lib/jenkins/pytorch/"

# IMAGE_NAME=nvcr.io/nvidia/pytorch:21.08-py3
# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:8733_ubuntu18.04_py3.6_pytorch_rocm4.5_internal_testing_warpsize_a1f2f40
# IMAGE_NAME=computecqe/pytorch:8863_warpsize32
# IMAGE_NAME=rocm/pytorch-private:SWDEV-navi
# IMAGE_NAME=computecqe/pytorch:rocm4.5_internal_testing_warpsize_mmelesse_pr_2
# IMAGE_NAME=rocm/pytorch-private:SWDEV-fixnosymbol
# IMAGE_NAME=rocm/pytorch-private:SWDEV-navihang
# IMAGE_NAME=rocm/pytorch-private:SWDEV-hangsols
# IMAGE_NAME=rocm/pytorch-private:SWDEV-printsleepkernel
# IMAGE_NAME=rocm/pytorch-private:SWDEV-printsleepkernel-rocm5.0
# IMAGE_NAME=computecqe/pytorch:warpsize32_build_8950
# IMAGE_NAME=rocm/pytorch-private:binomal_bug
# IMAGE_NAME=rocm/pytorch-private:9013_ubuntu18.04_py3.6_pytorch_rocm5.0_internal_testing_96a921b
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-5.0:19_ubuntu18.04_py3.7_pytorch_rocm5.0_internal_testing_6003aca
# IMAGE_NAME=rocm/pytorch-nightly:latest
IMAGE_NAME=rocm/pytorch:rocm4.5.2_ubuntu18.04_py3.8_pytorch_1.10.0
# IMAGE_NAME=rocm/pytorch-private:const_num_threads_128
# IMAGE_NAME=rocm/pytorch-private:const_num_threads_256

CONTAINER_NAME=pytorch

# start new container
CONTAINER_ID=$(drun -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
