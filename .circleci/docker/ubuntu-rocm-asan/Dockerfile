ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Set AMD gpu targets to build for
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}

# Install common dependencies (so that this step can be cached separately)
ARG EC2
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install rocm
RUN cd /opt && \
wget http://compute-artifactory.amd.com/artifactory/rocm-release-archive/afar-installation-tar/compute-rocm-dkms-asan-afar-318/rocm-asan-318-ubuntu.tar.bz2 && \
tar -xvjf rocm-asan-318-ubuntu.tar.bz2 && \
rm rocm-asan-318-ubuntu.tar.bz2 && \
ln -s rocm-asan-318 rocm
RUN apt-get update && apt-get install -y libnuma-dev libdrm-dev kmod wget
ENV LD_LIBRARY_PATH /opt/rocm/llvm/lib/clang/17.0.0/lib/linux
ENV CC "/opt/rocm/llvm/bin/clang"
ENV CXX "/opt/rocm/llvm/bin/clang++"
ENV LDSHARED "/opt/rocm/llvm/bin/clang --shared -fuse-ld=lld -ggdb -fsanitize=address -shared-libasan -lresolv"
ENV LDFLAGS "-fuse-ld=lld -ggdb -fsanitize=address -shared-libasan -lresolv"
ENV CFLAGS "-ggdb -fsanitize=address -shared-libasan"
ENV CXXFLAGS "-ggdb -fsanitize=address -shared-libasan"
ENV ASAN_OPTIONS detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
ENV PATH /opt/rocm/bin:$PATH
ENV PATH /opt/rocm/hcc/bin:$PATH
ENV PATH /opt/rocm/hip/bin:$PATH
ENV PATH /opt/rocm/opencl/bin:$PATH
ENV PATH /opt/rocm/llvm/bin:$PATH
ENV PYTORCH_ROCM_ARCH gfx90a:xnack+
ENV HSA_XNACK 1

# Install python and other packages (e.g., numpy, pytest)
# This also installs MKL, which magma uses.
ENV PATH /opt/conda/bin:$PATH
ARG ANACONDA_PYTHON_VERSION
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_python.sh install_python.sh
RUN bash ./install_python.sh && rm install_python.sh
RUN rm /opt/conda/requirements-ci.txt
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/conda/lib"

# Install rocm magma
COPY ./common/install_rocm_magma.sh install_rocm_magma.sh
RUN bash ./install_rocm_magma.sh
RUN rm install_rocm_magma.sh
ENV MAGMA_HOME /opt/rocm/magma
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# (optional) Install protobuf for ONNX
ARG PROTOBUF
COPY ./common/install_protobuf.sh install_protobuf.sh
RUN if [ -n "${PROTOBUF}" ]; then bash ./install_protobuf.sh; fi
RUN rm install_protobuf.sh
ENV INSTALLED_PROTOBUF ${PROTOBUF}

# (optional) Install database packages like LMDB and LevelDB
ARG DB
COPY ./common/install_db.sh install_db.sh
RUN if [ -n "${DB}" ]; then bash ./install_db.sh; fi
RUN rm install_db.sh
ENV INSTALLED_DB ${DB}

# (optional) Install vision packages like OpenCV and ffmpeg
ARG VISION
COPY ./common/install_vision.sh install_vision.sh
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh
ENV INSTALLED_VISION ${VISION}

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
COPY ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
COPY ./common/install_ninja.sh install_ninja.sh
RUN if [ -n "${NINJA_VERSION}" ]; then bash ./install_ninja.sh; fi
RUN rm install_ninja.sh

## Install ccache/sccache (do this last, so we get priority in PATH)
#COPY ./common/install_cache.sh install_cache.sh
#ENV PATH /opt/cache/bin:$PATH
#RUN bash ./install_cache.sh && rm install_cache.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
