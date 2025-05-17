#include <gtest/gtest.h>
#include "caffe2/core/hip/blob.h"
#include "caffe2/core/hip/context.h"
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/tensor.h"
#include "caffe2/operators/hip/utility_ops.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/hip/conversions.h"
#include "caffe2/utils/hip/math.h"

namespace caffe2 {

TEST(MathROCBLASTest, GemmNoTransNoTrans) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);

  Blob* blobX = ws.CreateBlob("X");
  Blob* blobW = ws.CreateBlob("W");
  Blob* blobY = ws.CreateBlob("Y");
  Blob* blobY_host = ws.CreateBlob("Y_host");

  vector<int> shapeX{5, 10};
  vector<int> shapeW{10, 6};
  vector<int> shapeY{5, 6};
  auto* tensorX = BlobGetMutableTensor(blobX, CUDA);
  tensorX->Resize(shapeX);
  auto* tensorW = BlobGetMutableTensor(blobW, CUDA);
  tensorW->Resize(shapeW);
  auto* tensorY = BlobGetMutableTensor(blobY, CUDA);
  tensorY->Resize(shapeY);
  auto* tensorY_host = BlobGetMutableTensor(blobY_host, CPU);
  tensorY_host->Resize(shapeY);

  EXPECT_EQ(tensorX->size(), 50);
  EXPECT_EQ(tensorW->size(), 60);
  EXPECT_EQ(tensorY->size(), 30);

  math::Set<float, CUDAContext>(
      tensorX->size(), 1, tensorX->mutable_data<float>(), &context);
  math::Set<float, CUDAContext>(
      tensorW->size(), 1, tensorW->mutable_data<float>(), &context);

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kOne,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kZero,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 10) << i;
  }

  // Test Accumulate
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kOne,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kPointFive,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 15) << i;
  }

  // Test Accumulate
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kPointFive,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kOne,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 20) << i;
  }
}

TEST(MathROCBLASTest, GemmNoTransTrans) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);

  Blob* blobX = ws.CreateBlob("X");
  Blob* blobW = ws.CreateBlob("W");
  Blob* blobY = ws.CreateBlob("Y");
  Blob* blobY_host = ws.CreateBlob("Y_host");

  vector<int> shapeX{5, 10};
  vector<int> shapeW{6, 10};
  vector<int> shapeY{5, 6};
  auto* tensorX = BlobGetMutableTensor(blobX, CUDA);
  tensorX->Resize(shapeX);
  auto* tensorW = BlobGetMutableTensor(blobW, CUDA);
  tensorW->Resize(shapeW);
  auto* tensorY = BlobGetMutableTensor(blobY, CUDA);
  tensorY->Resize(shapeY);
  auto* tensorY_host = BlobGetMutableTensor(blobY_host, CPU);
  tensorY_host->Resize(shapeY);

  EXPECT_EQ(tensorX->size(), 50);
  EXPECT_EQ(tensorW->size(), 60);
  EXPECT_EQ(tensorY->size(), 30);

  math::Set<float, CUDAContext>(
      tensorX->size(), 1, tensorX->mutable_data<float>(), &context);
  math::Set<float, CUDAContext>(
      tensorW->size(), 1, tensorW->mutable_data<float>(), &context);

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kOne,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kZero,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 10) << i;
  }

  // Test Accumulate
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kOne,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kPointFive,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 15) << i;
  }

  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kPointFive,
      tensorX->template data<float>(),
      tensorW->template data<float>(),
      kOne,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  EXPECT_EQ(tensorY_host->size(), 30);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 20) << i;
  }
}

TEST(MathROCBLASTest, GemvNoTrans) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);

  Blob* blobA = ws.CreateBlob("A");
  Blob* blobX = ws.CreateBlob("X");
  Blob* blobY = ws.CreateBlob("Y");
  Blob* blobY_host = ws.CreateBlob("Y_host");

  vector<int> shapeA{5, 10};
  vector<int> shapeX{10};
  vector<int> shapeY{5};
  auto* tensorA = BlobGetMutableTensor(blobA, CUDA);
  tensorA->Resize(shapeA);
  auto* tensorX = BlobGetMutableTensor(blobX, CUDA);
  tensorX->Resize(shapeX);
  auto* tensorY = BlobGetMutableTensor(blobY, CUDA);
  tensorY->Resize(shapeY);
  auto* tensorY_host = BlobGetMutableTensor(blobY_host, CPU);
  tensorY_host->Resize(shapeY);

  EXPECT_EQ(tensorA->size(), 50);
  EXPECT_EQ(tensorX->size(), 10);
  EXPECT_EQ(tensorY->size(), 5);
  math::Set<float, CUDAContext>(
      tensorA->size(), 1, tensorA->mutable_data<float>(), &context);
  math::Set<float, CUDAContext>(
      tensorX->size(), 1, tensorX->mutable_data<float>(), &context);

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      5,
      10,
      kOne,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kZero,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 10) << i;
  }

  // Test Accumulate
  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      5,
      10,
      kOne,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kPointFive,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 15) << i;
  }

  // Test Accumulate
  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      5,
      10,
      kPointFive,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kOne,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 20) << i;
  }
}

TEST(MathROCBLASTest, GemvTrans) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);

  Blob* blobA = ws.CreateBlob("A");
  Blob* blobX = ws.CreateBlob("X");
  Blob* blobY = ws.CreateBlob("Y");
  Blob* blobY_host = ws.CreateBlob("Y_host");

  vector<int> shapeA{6, 10};
  vector<int> shapeX{6};
  vector<int> shapeY{10};
  auto* tensorA = BlobGetMutableTensor(blobA, CUDA);
  tensorA->Resize(shapeA);
  auto* tensorX = BlobGetMutableTensor(blobX, CUDA);
  tensorX->Resize(shapeX);
  auto* tensorY = BlobGetMutableTensor(blobY, CUDA);
  tensorY->Resize(shapeY);
  auto* tensorY_host = BlobGetMutableTensor(blobY_host, CPU);
  tensorY_host->Resize(shapeY);

  EXPECT_EQ(tensorA->size(), 60);
  EXPECT_EQ(tensorX->size(), 6);
  EXPECT_EQ(tensorY->size(), 10);
  math::Set<float, CUDAContext>(
      tensorA->size(), 1, tensorA->mutable_data<float>(), &context);
  math::Set<float, CUDAContext>(
      tensorX->size(), 1, tensorX->mutable_data<float>(), &context);

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemv<float, CUDAContext>(
      CblasTrans,
      6,
      10,
      kOne,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kZero,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 6) << i;
  }

  // Test Accumulate
  math::Gemv<float, CUDAContext>(
      CblasTrans,
      6,
      10,
      kOne,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kPointFive,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 9) << i;
  }

  // Test Accumulate
  math::Gemv<float, CUDAContext>(
      CblasTrans,
      6,
      10,
      kPointFive,
      tensorA->data<float>(),
      tensorX->data<float>(),
      kOne,
      tensorY->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();
  tensorY_host->CopyFrom(*tensorY);
  for (int i = 0; i < tensorY_host->size(); ++i) {
    TORCH_CHECK_EQ(tensorY_host->data<float>()[i], 12) << i;
  }
}
} // namespace caffe2
