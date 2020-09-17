![AMD Logo](/Installation/amdblack.jpg)


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

1.  Run the following code to ensure that your system is up to date:

```{=html}
<!-- -->
```
    sudo apt update

    sudo apt dist-upgrade

    sudo apt install libnuma-dev

    sudo reboot 

2.  Add the ROCm apt repository.

For Debian-based systems like Ubuntu, configure the Debian ROCm repository as follows:

**Note**: The public key has changed to reflect the new location. You must update to the new location as the old key will be removed in a
future release.

-   Old Key: <http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key>
-   New Key: <http://repo.radeon.com/rocm/rocm.gpg.key>

```{=html}
<!-- -->
```
    wget -q -O - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add
the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

    e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key

3.  Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

```{=html}
<!-- -->
```
    sudo apt update

    sudo apt install rocm-dkms && sudo reboot

4.  Set permissions. To access the GPU, you must be a user in the video and render groups. Ensure your user account is a member of the video
    and render groups prior to using ROCm. To identify the groups you are a member of, use the following command:

```{=html}
<!-- -->
```
    groups

5.  To add your user to the video and render groups, use the following command with the sudo password:

```{=html}
<!-- -->
```
    sudo usermod -a -G video $LOGNAME

    sudo usermod -a -G render $LOGNAME

6.  By default, you must add any future users to the video and render groups. To add future users to the video and render groups, run the
    following command:

```{=html}
<!-- -->
```
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf

    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

    echo 'EXTRA_GROUPS=render' | sudo tee -a /etc/adduser.conf

7.  Restart the system.
8.  After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs
    listed by both commands, the installation is considered successful.

```{=html}
<!-- -->
```
    /opt/rocm/bin/rocminfo
    /opt/rocm/opencl/bin/clinfo

Note: To run the ROCm programs, add the ROCm binaries in your PATH.

    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh




