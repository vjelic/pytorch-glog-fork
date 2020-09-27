![AMD Logo](/Installation/amdblack.jpg)


# AMD ROCm Installation Guide v3.7

## Install AMD ROCm


-   [Deploying ROCm](#deploying-rocm)

-   [Prerequisites](#prerequisites-1)

-   [Install ROCm on Supported Operating Systems](#supported-operating-systems)

     -   [Ubuntu](#ubuntu)
     -   [CentOS RHEL](#centos-rhel)
     -   [SLES 15 Service Pack 1](#sles-15-service-pack-1)
    
-   [ROCm Installation Known Issues and
    Workarounds](#rocm-installation-known-issues-and-workarounds)


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

```
    sudo apt update

    sudo apt dist-upgrade

    sudo apt install libnuma-dev

    sudo reboot 
    
 ```
 
2.  Add the ROCm apt repository.

For Debian-based systems like Ubuntu, configure the Debian ROCm repository as follows:

**Note**: The public key has changed to reflect the new location. You must update to the new location as the old key will be removed in a
future release.

-   Old Key: <http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key>
-   New Key: <http://repo.radeon.com/rocm/rocm.gpg.key>


```
    wget -q -O - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

```

The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add
the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

    e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key

3.  Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

```
    sudo apt update

    sudo apt install rocm-dkms && sudo reboot
 ```

4.  Set permissions. To access the GPU, you must be a user in the video and render groups. Ensure your user account is a member of the video
    and render groups prior to using ROCm. To identify the groups you are a member of, use the following command:

```
    groups
    
 ```

5.  To add your user to the video and render groups, use the following command with the sudo password:

```
    sudo usermod -a -G video $LOGNAME

    sudo usermod -a -G render $LOGNAME
    
```

6.  By default, you must add any future users to the video and render groups. To add future users to the video and render groups, run the
    following command:

```
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf

    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

    echo 'EXTRA_GROUPS=render' | sudo tee -a /etc/adduser.conf
 
 ```

7.  Restart the system.

8.  After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs
    listed by both commands, the installation is considered successful.

```
    /opt/rocm/bin/rocminfo
    /opt/rocm/opencl/bin/clinfo
    
 ```

**Note**: To run the ROCm programs, add the ROCm binaries in your PATH.

```

    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh

```

#### Uninstalling ROCm Packages from Ubuntu

To uninstall the ROCm packages from Ubuntu 16.04.6 or Ubuntu 18.04.4, run the following command:

    sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils && sudo reboot

#### Installing Development Packages for Cross Compilation

It is recommended that you develop and test development packages on different systems. For example, some development or build systems may
not have an AMD GPU installed. In this scenario, you must avoid installing the ROCk kernel driver on the development system.

Instead, install the following development subset of packages:

    sudo apt update
    sudo apt install rocm-dev

**Note**: To execute ROCm enabled applications, you must install the full ROCm driver stack on your system.

#### Using Debian-based ROCm with Upstream Kernel Drivers

You can install the ROCm user-level software without installing the AMD\'s custom ROCk kernel driver. To use the upstream kernels, run the
following commands instead of installing rocm-dkms:

    sudo apt update   
    sudo apt install rocm-dev 
    echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules


### CentOS RHEL

#### CentOS v7.7/RHEL v7.8 and CentOS/RHEL 8.1

This section describes how to install ROCm on supported RPM-based systems such as CentOS v7.7/RHEL v7.8 and CentOS/RHEL v8.1.

#### Preparing RHEL for Installation

RHEL is a subscription-based operating system. You must enable the external repositories to install on the devtoolset-7 environment and the
dkms support files.

**Note**: The following steps do not apply to the CentOS installation.

1.  The subscription for RHEL must be enabled and attached to a pool ID. See the Obtaining an RHEL image and license page for instructions on
    registering your system with the RHEL subscription server and attaching to a pool id.
    
2.  Enable the following repositories for RHEL v7.x:

```
    sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms 
    sudo subscription-manager repos --enable rhel-7-server-optional-rpms
    sudo subscription-manager repos --enable rhel-7-server-extras-rpms

```

3.  Enable additional repositories by downloading and installing the epel-release-latest-7/epel-release-latest-8 repository RPM:

```
    sudo rpm -ivh <repo>
```

For more details,

-   For RHEL v7.x, see
    <https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm>
    
-   For RHEL v8.x, see
    <https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm>
    

4.  Install and set up Devtoolset-7.

**Note**: Devtoolset is not required for CentOS/RHEL v8.x

To setup the Devtoolset-7 environment, follow the instructions on this page: <https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/>

**Note**: devtoolset-7 is a software collections package and is not supported by AMD.

### Installing CentOS v7.7/v8.1 for DKMS

Use the dkms tool to install the kernel drivers on CentOS/RHEL:

    sudo yum install -y epel-release
    sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`


#### Installing ROCm 

To install ROCm on your system, follow the instructions below:

1.  Delete the previous versions of ROCm before installing the latest version.

2.  Create a /etc/yum.repos.d/rocm.repo file with the following contents:

     -   CentOS/RHEL 7.x : <http://repo.radeon.com/rocm/yum/rpm>
     -   CentOS/RHEL 8.x : <http://repo.radeon.com/rocm/centos8/rpm>

```
    [ROCm] 
    name=ROCm
    baseurl=http://repo.radeon.com/rocm/yum/rpm
    enabled=1
    gpgcheck=1
    gpgkey=http://repo.radeon.com/rocm/rocm.gpg.key

```

**Note**: The URL of the repository must point to the location of the repositories' repodata database.

3.  Install ROCm components using the following command:

**Note**: This step is applicable only for CentOS/RHEL v8.1 and is not required for v7.8.

```
    sudo yum install rocm-dkms && sudo reboot
    
```

4.  Restart the system. The rock-dkms component is installed and the /dev/kfd device is now available.

5.  Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior
    to using ROCm. To identify the groups you are a member of, use the following command:

```
    groups
    
```

6.  To add your user to the video group, use the following command with the sudo password:

```
    sudo usermod -a -G video $LOGNAME
 ```

7.  By default, add any future users to the video group. Run the following command to add users to the video group:

```
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
```

**Note**: Before updating to the latest version of the operating system, delete the ROCm packages to avoid DKMS-related issues.

8.  Restart the system.

9.  Test the ROCm installation.

#### Testing the ROCm Installation

After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed, you
are good to go!

    /opt/rocm/bin/rocminfo
    /opt/rocm/opencl/bin/clinfo

**Note**: Add the ROCm binaries in your PATH for easy implementation of the ROCm programs.

    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh

#### Compiling Applications Using HCC, HIP, and Other ROCm Software

To compile applications or samples, run the following command to use gcc-7.2 provided by the devtoolset-7 environment:

    scl enable devtoolset-7 bash
    

#### Uninstalling ROCm from CentOS/RHEL

To uninstall the ROCm packages, run the following command:

    sudo yum autoremove rocm-opencl rocm-dkms rock-dkms

#### Installing Development Packages for Cross Compilation

You can develop and test ROCm packages on different systems. For example, some development or build systems may not have an AMD GPU
installed. In this scenario, you can avoid installing the ROCm kernel driver on your development system. Instead, install the following
development subset of packages:

    sudo yum install rocm-dev

**Note**: To execute ROCm-enabled applications, you will require a system installed with the full ROCm driver stack.


#### Using ROCm with Upstream Kernel Drivers

You can install ROCm user-level software without installing AMD\'s custom ROCk kernel driver. To use the upstream kernel drivers, run the
following commands

    sudo yum install rocm-dev
    echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules  
    sudo reboot

**Note**: You can use this command instead of installing rocm-dkms.

**Note**: Ensure you restart the system after ROCm installation.


### SLES 15 Service Pack 1

The following section tells you how to perform an install and uninstall ROCm on SLES 15 SP 1.

**Installation**

1.  Install the \"dkms\" package.

```
    sudo SUSEConnect --product PackageHub/15.1/x86_64
    sudo zypper install dkms
```
2.  Add the ROCm repo.

```
    sudo zypper clean â€“all
    sudo zypper addrepo http://repo.radeon.com/rocm/zyp/zypper/ rocm
    sudo zypper ref
    sudo rpm --import http://repo.radeon.com/rocm/rocm.gpg.key
    sudo zypper --gpg-auto-import-keys install rocm-dkms
    sudo reboot
```
3.  Run the following command once

```
    cat <<EOF | sudo tee /etc/modprobe.d/10-unsupported-modules.conf
    allow_unsupported_modules 1
    EOF
    sudo modprobe amdgpu
```
4.  Verify the ROCm installation.

5.  Run /opt/rocm/bin/rocminfo and /opt/rocm/opencl/bin/clinfo commands to list the GPUs and verify that the ROCm installation is
    successful.
    
6.  Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

    groups

7.  To add your user to the video group, use the following command with the sudo password:

```
    sudo usermod -a -G video $LOGNAME
```
8.  By default, add any future users to the video group. Run the following command to add users to the video group:

```
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
```
9.  Restart the system.

10. Test the basic ROCm installation.

11. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs
    listed by both commands, the installation is considered successful.

```
    /opt/rocm/bin/rocminfo
    /opt/rocm/opencl/bin/clinfo
```
**Note**: To run the ROCm programs more efficiently, add the ROCm binaries in your PATH.

     echo \'export
     PATH=\$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin\'\|sudo
     tee -a /etc/profile.d/rocm.sh

#### Uninstallation

To uninstall, use the following command:

    sudo zypper remove rocm-opencl rocm-dkms rock-dkms

**Note**: Ensure all other installed packages/components are removed. 

**Note**: Ensure all the content in the /opt/rocm directory is completely removed. If the command does not remove all the ROCm components/packages, ensure
you remove them individually.


## ROCm Installation Known Issues and Workarounds

### Closed source components

The ROCm platform relies on some closed source components to provide functionalities like HSA image support. These components are only
available through the ROCm repositories, and they may be deprecated or become open source components in the future. These components are made
available in the following packages:

     -   hsa-ext-rocr-dev
