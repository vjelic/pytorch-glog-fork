#include <ATen/miopen/Exceptions.h>
#include <ATen/miopen/Handle.h>
#include <c10/hip/HIPStream.h>

#include <memory>
#include <mutex>

namespace at { namespace native {

namespace {

struct Handle {
  miopenHandle_t handle;
  Handle() : handle(NULL) {
    MIOPEN_CHECK(miopenCreate(&handle));
  }
  ~Handle() {
    if (handle) {
      miopenDestroy(handle);
    }
  }
};

std::mutex mutex;

std::once_flag flag;

std::unique_ptr<Handle> handle;

};

MIOpenHandle getMiopenHandle()
{
  std::call_once(flag, [](){ handle.reset(new Handle); });
  std::unique_lock<std::mutex> lock(mutex);
  MIOPEN_CHECK(miopenSetStream(handle->handle, at::hip::getCurrentHIPStream()));
  return MIOpenHandle(std::move(lock), handle->handle);
}

}} // namespace at::native
