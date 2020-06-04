#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    // See Note [ATen preprocessor philosophy]
  std::tuple<Tensor, Tensor> miopen_max_pooling(
      const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, 
      IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    AT_ERROR("miopen_pooling: ATen not compiled with MIOpen support");
  }

  Tensor miopen_max_pooling_backward(
      const Tensor& grad_output, const at::Tensor& input_t, IntArrayRef kernel_size,
      IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices_t) {
    AT_ERROR("miopen_pooling_backward: ATen not compiled with MIOpen support");
  }                                  

}}

#else //AT_ROCM_ENABLED

#include <THH/THH.h>

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

#include <iostream>

namespace at { namespace native {

  //Calculate Pooling output shape.
  static std::vector<int64_t> pooling_output_size(
      IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef padding,
      IntArrayRef stride, IntArrayRef dilation, bool ceil_mode)
  {
      auto dim = input_size.size();
      std::vector<int64_t> output_size(dim);
      int hw_start_dim = 2;
      if (dim == 3) {
        output_size[0] = input_size[0];
        hw_start_dim = 1;
      } else {
        output_size[0] = input_size[0];
        output_size[1] = input_size[1];
      }
      for (size_t d = hw_start_dim; d < dim ; ++d) {
          output_size[d] = ((input_size[d] + 2 * padding[d - hw_start_dim] - dilation[d - hw_start_dim] * (kernel_size[d - hw_start_dim] - 1)
                                         - 1 + (ceil_mode ? stride[d - hw_start_dim] : 0))/ stride[d - hw_start_dim] + 1);
      }
      return output_size;
  }

  std::vector<int> get_miopen_tensor_sizes(std::vector<int64_t> input_size) {
    if (input_size.size() == 3) {
        std::vector<int> mio_dims;
        mio_dims.push_back(input_size[0]);
        mio_dims.push_back(1);
        mio_dims.push_back(input_size[1]);
        mio_dims.push_back(input_size[2]);
        return mio_dims;
    } else {
        std::vector<int> mio_dims(begin(input_size), end(input_size));
        return mio_dims;
    }
  }

  //Perform miopen max pool operation.
  std::tuple<at::Tensor, at::Tensor> miopen_max_pooling(
      const Tensor& input_t, IntArrayRef kernel_size, IntArrayRef stride,
      IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
  {
      TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4), "non-empty 3D or 4D (batch-mode) tensor expected for input.");
      TORCH_CHECK(kernel_size.size() == 2, "Currently MIOpen supports 2D Pooling.");
      //TensorArg input { input_t, "input", 0};
      setMIOpenStreamToCurrent();
      CheckedFrom c = "miopen_pooling";

      TensorArg input {input_t, "input", 0};
      checkAllDefined(c, {input});
      
      //create output and indices tensor.
      auto output_t = at::empty(pooling_output_size(input->sizes(), kernel_size, padding, stride, dilation, ceil_mode), input->options());
      TensorArg output { output_t, "result", 1 };

      auto indices_t = at::empty(output->sizes(), output->options().dtype(kLong));
      TensorArg indices {indices_t, "indices", 2};

      //Pooling mode.
      miopenPoolingMode_t mode = miopenPoolingMax;
      auto handle = getMiopenHandle();
      auto datatype = getMiopenDataType(*input);

      //Input and output descriptors.
      //TensorDescriptor idesc{ *input, 4}; //input descriptor
      //TensorDescriptor odesc{ *output, 4}; //output descriptor
      std::vector<int> input_dims = get_miopen_tensor_sizes(input->sizes().vec());
      std::vector<int> output_dims = get_miopen_tensor_sizes(output->sizes().vec());
      miopenTensorDescriptor_t idesc, odesc;
      miopenCreateTensorDescriptor(&idesc);
      miopenCreateTensorDescriptor(&odesc);
      miopenSetTensorDescriptor(idesc, datatype, input_dims.size(), input_dims.data(), nullptr);
      miopenSetTensorDescriptor(odesc, datatype, output_dims.size(), output_dims.data(), nullptr);
      
      //Pooling Descriptor.    
      miopenPoolingDescriptor_t pdesc;
      miopenCreatePoolingDescriptor(&pdesc);
      MIOPEN_CHECK(miopenSet2dPoolingDescriptor(pdesc, mode, kernel_size[0], kernel_size[1], padding[0], padding[1], stride[0], stride[1]));

      //Get pooling workspace size. 
      size_t ws_size;
      miopenIndexType_t index_type = miopenIndexUint64;
      miopenSetPoolingIndexType(pdesc, index_type);
      miopenPoolingWorkspaceIndexMode_t ws_index_mode = miopenPoolingWorkspaceIndexImage;
      miopenSetPoolingWorkSpaceIndexMode(pdesc, ws_index_mode); 
      //MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, odesc.desc(), &ws_size));
      MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, odesc, &ws_size));

      Constant one(datatype, 1);
      Constant zero(datatype, 0);
     
      //Run miopen pooling forward and return the indices and the output tensor.
      //MIOPEN_CHECK(miopenPoolingForward(handle, pdesc, &one, 
      //                    idesc.desc(), input->data_ptr(),
      //                    &zero, odesc.desc(), output->data_ptr(),
      //                    true, indices->data_ptr(), ws_size));
      MIOPEN_CHECK(miopenPoolingForward(handle, pdesc, &one,
                            idesc, input->data_ptr(), &zero,
                            odesc, output->data_ptr(),
                            true, indices->data_ptr(), ws_size));

      return std::tuple<at::Tensor, at::Tensor>{output_t, indices_t};
  } 

  //Perform miopen maxpooling backwards.
  Tensor miopen_max_pooling_backward(
      const Tensor& grad_output_t, const Tensor& input_t, IntArrayRef kernel_size, IntArrayRef stride,
      IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices_t)
  {
      TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4), "non-empty 3D or 4D (batch-mode) tensor expected for input.");
      TORCH_CHECK(kernel_size.size() == 2, "Currently MIOpen supports 2D Pooling.");
      TensorArg grad_output {grad_output_t, "grad_output", 1},
                input {input_t, "input", 2},
                indices {indices_t, "indices", 3};
      
      CheckedFrom c = "miopen_max_pooling_backwards";
      setMIOpenStreamToCurrent();
      
      checkAllDefined(c, {grad_output, input, indices});
      checkAllSameGPU(c, {grad_output, input, indices});

      auto grad_input_t = at::empty(input->sizes(), input->options());
      auto output_t = at::empty(pooling_output_size(input->sizes(), kernel_size, padding, stride, dilation, ceil_mode), grad_output->options());

      TensorArg grad_input {grad_input_t, "grad_input", 0};
      TensorArg output {output_t, "output", 4};
         
      auto handle = getMiopenHandle();
      auto datatype = getMiopenDataType(*input);

      //TensorDescriptor godesc{ *grad_output, 4};
      //TensorDescriptor odesc {*output, 4};
      //TensorDescriptor idesc{ *input, 4};
      std::vector<int> input_dims = get_miopen_tensor_sizes(input->sizes().vec());
      std::vector<int> output_dims = get_miopen_tensor_sizes(output->sizes().vec());
      std::vector<int> grad_out_dims = get_miopen_tensor_sizes(grad_output->sizes().vec());
      miopenTensorDescriptor_t idesc, odesc, godesc;
      miopenCreateTensorDescriptor(&idesc);
      miopenCreateTensorDescriptor(&odesc);
      miopenCreateTensorDescriptor(&godesc);
      miopenSetTensorDescriptor(idesc, datatype, input_dims.size(), input_dims.data(), nullptr);
      miopenSetTensorDescriptor(odesc, datatype, output_dims.size(), output_dims.data(), nullptr);
      miopenSetTensorDescriptor(godesc, datatype, grad_out_dims.size(), grad_out_dims.data(), nullptr);

      //Pooling mode.
      miopenPoolingMode_t mode = miopenPoolingMax;
      
      //Pooling descriptor.
      miopenPoolingDescriptor_t pdesc;
      miopenCreatePoolingDescriptor(&pdesc);
      MIOPEN_CHECK(miopenSet2dPoolingDescriptor(pdesc, mode, kernel_size[0], kernel_size[1], padding[0], padding[1], stride[0], stride[1]));

      //Get pooling workspace size. 
      size_t ws_size;
      miopenIndexType_t index_type = miopenIndexUint64;
      miopenSetPoolingIndexType(pdesc, index_type);
      miopenPoolingWorkspaceIndexMode_t ws_index_mode = miopenPoolingWorkspaceIndexImage;
      miopenSetPoolingWorkSpaceIndexMode(pdesc, ws_index_mode);
      //MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, godesc.desc(), &ws_size));
      MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, godesc, &ws_size));

      Constant one(datatype, 1);
      Constant zero(datatype, 0);

      //Run MIOpen backward Pooling.
      //MIOPEN_CHECK(miopenPoolingBackward(handle, pdesc, &one,
      //                odesc.desc(), output->data_ptr(),
      //                godesc.desc(), grad_output->data_ptr(),
      //                idesc.desc(), input->data_ptr(), &zero, 
      //                idesc.desc(), grad_input->data_ptr(), indices->data_ptr()                 
      //                ));
      MIOPEN_CHECK(miopenPoolingBackward(handle, pdesc,&one,
                        odesc, output->data_ptr(),
                        godesc, grad_output->data_ptr(),
                        idesc, input->data_ptr(), &zero,
                        idesc, grad_input->data_ptr(), indices->data_ptr()));
              
      return grad_input_t;
  }

}} //namespace at::native

#endif
