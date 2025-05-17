# ---[ CPU files.
set(NOMNI_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/Representations/NeuralNet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_util.cc
)
set(NOMNI_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/AlgorithmsTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/BinaryMatchImplTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/GraphTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/MatchTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralNetTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/SubgraphMatcherTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/TarjansImplTest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/TopoSortTest.cc
)
set(NOMNI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

list(APPEND Caffe2_CPU_SRCS ${NOMNI_SRCS})
list(APPEND Caffe2_CPU_INCLUDE ${NOMNI_INCLUDE_DIR})
list(APPEND Caffe2_GPU_INCLUDE ${NOMNI_INCLUDE_DIR})
list(APPEND Caffe2_CPU_TEST_SRCS ${NOMNI_TEST_SRCS})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN "*.h")

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_INCLUDE ${Caffe2_CPU_INCLUDE} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
if(USE_TENSORRT)
set(Caffe2_GPU_INCLUDE ${Caffe2_CPU_INCLUDE} PARENT_SCOPE)
endif()
