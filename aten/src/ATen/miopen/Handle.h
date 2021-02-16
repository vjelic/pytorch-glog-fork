#pragma once

#include <mutex>

#include <ATen/miopen/miopen-wrapper.h>

namespace at { namespace native {

// RAII wrapper for all calls to MIOpen with a MIOpen handle argument.
class MIOpenHandle {
  public:
    // Takes ownership of the lock to access MIOpen using handle.
    MIOpenHandle(std::unique_lock<std::mutex> lock, miopenHandle_t handle)
      : lock_(std::move(lock)), handle_(handle) {}

    // Returns MIOpen handle. To be passed directly to MIOpen APIs, don't keep a copy.
    miopenHandle_t handle() const { return handle_; }

  private:
    std::unique_lock<std::mutex> lock_;
    miopenHandle_t handle_;  // Not owned.
};

MIOpenHandle getMiopenHandle();

}} // namespace
