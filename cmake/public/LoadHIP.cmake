set(PYTORCH_FOUND_HIP FALSE)

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH /opt/rocm)
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()

if(NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
  set(PYTORCH_ROCM_ARCH gfx803;gfx900;gfx906;gfx908)
else()
  set(PYTORCH_ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
endif()

# Add HIP to the CMake prefix path, for find_package later
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
endmacro()

# Find the HIP Package
find_package_and_print_version(hip)

if(hip_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)

  # Find ROCM version for checks
  file(READ "${ROCM_PATH}/.info/version-dev" ROCM_VERSION_DEV_RAW)
  string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)-.*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})
  if(ROCM_VERSION_DEV_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
  endif()
  message("\n***** ROCm version from ${ROCM_PATH}/.info/version-dev ****\n")
  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")

  message("\n***** Library versions from dpkg *****\n")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-libs COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hsakmt-roct COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocr-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip_base COMMAND awk "{print $2 \" VERSION: \" $3}")

  message("\n***** Library versions from cmake find_package *****\n")

  #set(hip_DIR ${HIP_PATH}/lib/cmake/hip)
  #set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
  #set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
  #set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
  #set(rocrand_DIR ${ROCRAND_PATH}/lib/cmake/rocrand)
  #set(hiprand_DIR ${HIPRAND_PATH}/lib/cmake/hiprand)
  #set(rocblas_DIR ${ROCBLAS_PATH}/lib/cmake/rocblas)
  #set(miopen_DIR ${MIOPEN_PATH}/lib/cmake/miopen)
  #set(rocfft_DIR ${ROCFFT_PATH}/lib/cmake/rocfft)
  #set(hipfft_DIR ${HIPFFT_PATH}/lib/cmake/hipfft)
  #set(hipsparse_DIR ${HIPSPARSE_PATH}/lib/cmake/hipsparse)
  #set(rccl_DIR ${RCCL_PATH}/lib/cmake/rccl)
  #set(rocprim_DIR ${ROCPRIM_PATH}/lib/cmake/rocprim)
  #set(hipcub_DIR ${HIPCUB_PATH}/lib/cmake/hipcub)
  #set(rocthrust_DIR ${ROCTHRUST_PATH}/lib/cmake/rocthrust)

  find_package_and_print_version(hip REQUIRED)
  find_package_and_print_version(hsa-runtime64 REQUIRED)
  find_package_and_print_version(amd_comgr REQUIRED)
  find_package_and_print_version(rocrand REQUIRED)
  find_package_and_print_version(hiprand REQUIRED)
  find_package_and_print_version(rocblas REQUIRED)
  find_package_and_print_version(miopen REQUIRED)
  find_package_and_print_version(rocfft REQUIRED)
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "4.1.0")
    find_package_and_print_version(hipfft REQUIRED)
  endif()
  find_package_and_print_version(hipsparse REQUIRED)
  find_package_and_print_version(rccl)
  find_package_and_print_version(rocprim REQUIRED)
  find_package_and_print_version(hipcub REQUIRED)
  find_package_and_print_version(rocthrust REQUIRED)

  #if(HIP_COMPILER STREQUAL clang)
  #  set(hip_library_name amdhip64)
  #else()
  #  set(hip_library_name hip_hcc)
  #endif()
  #message("HIP library name: ${hip_library_name}")

  ## TODO: hip_hcc has an interface include flag "-hc" which is only
  ## recognizable by hcc, but not gcc and clang. Right now in our
  ## setup, hcc is only used for linking, but it should be used to
  ## compile the *_hip.cc files as well.
  #find_library(PYTORCH_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${HIP_PATH}/lib)
  ## TODO: miopen_LIBRARIES should return fullpath to the library file,
  ## however currently it's just the lib name
  #find_library(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS ${MIOPEN_PATH}/lib)
  ## TODO: rccl_LIBRARIES should return fullpath to the library file,
  ## however currently it's just the lib name
  #find_library(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES} HINTS ${RCCL_PATH}/lib)
  ## hiprtc is part of HIP
  #find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
  ## roctx is part of roctracer
  #find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCTRACER_PATH}/lib)
  #set(roctracer_INCLUDE_DIRS ${ROCTRACER_PATH}/include)
endif()
