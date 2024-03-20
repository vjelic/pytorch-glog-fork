if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_TARBALL_TAG "0.4b")
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "6.0.0")
    set(__AOTRITON_TARBALL_PLATFORM "rocm6")
  else()
    set(__AOTRITON_TARBALL_PLATFORM "rocm5")
  endif()
  string(CONCAT __AOTRITON_TARBALL_URL
    "https://github.com/ROCm/aotriton/releases/download/"
    "${__AOTRITON_TARBALL_TAG}/"
    "aotriton-${__AOTRITON_TARBALL_TAG}"
    "-manylinux2014_x86_64-"
    "${__AOTRITON_TARBALL_PLATFORM}.tar.bz2")
  ExternalProject_Add(aotriton_external
    URL ${__AOTRITON_TARBALL_URL}
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/lib/libaotriton_v2.a
  )
  set(AOTRITON_FOUND TRUE)
  add_library(__caffe2_aotriton INTERFACE)
  add_dependencies(__caffe2_aotriton aotriton_external)
  target_link_libraries(__caffe2_aotriton INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/lib/libaotriton_v2.a)
  target_include_directories(__caffe2_aotriton INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/include)
endif() # __AOTRITON_INCLUDED
