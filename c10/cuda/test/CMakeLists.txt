# ---[ Test binaries.

set(C10_CUDA_ALL_TEST_FILES
    impl/CUDATest.cpp
)
if(USE_ROCM)
set(C10_LIB_NAME "c10_hip")
else()
set(C10_LIB_NAME "c10_cuda")
endif()

if(BUILD_TEST)
  foreach(test_src ${C10_CUDA_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_cuda_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} ${C10_LIB_NAME} gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()
