


# AMD ROCm Installation Guide v3.7

## Install AMD ROCm


-   [Deploying ROCm](#deploying-rocm)

-   [Prerequisites](#prerequisites-1)

-   [Supported Operating Systems](#supported-operating-systems)

    > -   [Ubuntu](#ubuntu)
    > -   [CentOS RHEL](#centos-rhel)
    > -   [SLES 15 Service Pack 1](#sles-15-service-pack-1)
    
-   [ROCm Installation Known Issues and
    Workarounds](#rocm-installation-known-issues-and-workarounds)

-   [Getting the ROCm Source Code](#getting-the-rocm-source-code)


## Deploying ROCm 

AMD hosts both Debian and RPM repositories for the ROCm v3.x packages.

The following directions show how to install ROCm on supported Debian-based systems such as Ubuntu 18.04.x

**Note**: These directions may not work as written on unsupported Debian-based distributions. For example, newer versions of Ubuntu may
not be compatible with the rock-dkms kernel driver. In this case, you can exclude the rocm-dkms and rock-dkms packages.


## Prerequisites

In this release, AMD ROCm extends support to Ubuntu 20.04, including dual kernel.

The AMD ROCm platform is designed to support the following operating systems:

-   Ubuntu 20.04 (5.4 and 5.6-oem) and 18.04.4 (Kernel 5.3)
-   CentOS 7.8 & RHEL 7.8 (Kernel 3.10.0-1127) (Using devtoolset-7
    runtime support)
-   CentOS 8.2 & RHEL 8.2 (Kernel 4.18.0 ) (devtoolset is not required)
-   SLES 15 SP1

### FRESH INSTALLATION OF AMD ROCm V3.7 RECOMMENDED

A fresh and clean installation of AMD ROCm v3.7 is recommended. An upgrade from previous releases to AMD ROCm v3.7 is not supported.

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a
fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.

**Note**: *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.

-   For ROCm v3.5 and releases thereafter, the *clinfo* path is changed
    to - */opt/rocm/opencl/bin/clinfo*.
    
-   For ROCm v3.3 and older releases, the *clinfo* path remains unchanged - */opt/rocm/opencl/bin/x86\_64/clinfo*.

    
## Supported Operating Systems

### Ubuntu

**Installing a ROCm Package from a Debian Repository**

To install from a Debian Repository:



