#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c10/hip/HIPGuard.h>
#include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/operators/hip/operator_fallback_gpu.h"
#include "caffe2/python/pybind_state_registry.h"

namespace caffe2 {
namespace python {

REGISTER_CUDA_OPERATOR(Python, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(PythonGradient, GPUFallbackOp);

REGISTER_CUDA_OPERATOR(PythonDLPack, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(PythonDLPackGradient, GPUFallbackOp);

REGISTER_BLOB_FEEDER(CUDA, TensorFeeder<CUDAContext>);

namespace py = pybind11;

void addCUDAGlobalMethods(py::module& m) {
  m.def("num_cuda_devices", &NumCudaDevices);
  m.def("get_cuda_version", &CudaVersion);
  m.def("get_miopen_version", &miopenCompiledVersion);
  m.def("get_gpu_memory_info", [](int device_id) {
    CUDAGuard guard(device_id);
    size_t device_free, device_total;
    CUDA_CHECK(hipMemGetInfo(&device_free, &device_total));
    return std::pair<size_t, size_t>{device_free, device_total};
  });
  m.def("get_cuda_peer_access_pattern", []() {
    std::vector<std::vector<bool>> pattern;
    CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&pattern));
    return pattern;
  });
  m.def("get_device_properties", [](int deviceid) {
    auto& prop = GetDeviceProperty(deviceid);
    std::map<std::string, py::object> obj;
    obj["name"] = py::cast(prop.name);
    obj["major"] = py::cast(prop.major);
    obj["minor"] = py::cast(prop.minor);
    obj["totalGlobalMem"] = py::cast(prop.totalGlobalMem);
    return obj;
  });
};

void addCUDAObjectMethods(py::module& m) {
  py::class_<DLPackWrapper<CUDAContext>>(m, "DLPackTensorCUDA")
      .def_property_readonly(
          "data",
          [](DLPackWrapper<CUDAContext>* t) -> py::object {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                PROTO_CUDA,
                "Expected CUDA device option for CUDA tensor");

            return t->data();
          },
          "Return DLPack tensor with tensor's data.")
      .def(
          "feed",
          [](DLPackWrapper<CUDAContext>* t, py::object obj) {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                PROTO_CUDA,
                "Expected CUDA device option for CUDA tensor");
            t->feed(obj);
          },
          "Copy data from given DLPack tensor into this tensor.")
      .def_property_readonly(
          "_shape",
          [](const DLPackWrapper<CUDAContext>& t) { return t.tensor->sizes(); })
      .def(
          "_reshape",
          [](DLPackWrapper<CUDAContext>* t, std::vector<int64_t> dims) {
            t->tensor->Resize(dims);
          });
}

PYBIND11_MODULE(caffe2_pybind11_state_cuda, m) {
  m.doc() = "pybind11 stateful interface to Caffe2 workspaces - GPU edition";

  addGlobalMethods(m);
  addCUDAGlobalMethods(m);
  addObjectMethods(m);
  addCUDAObjectMethods(m);
  for (const auto& addition : PybindAdditionRegistry()->Keys()) {
    PybindAdditionRegistry()->Create(addition, m);
  }
}
} // namespace python
} // namespace caffe2
