// =============================================================================
// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/contrib/periodic_resample/kernels/periodic_resample_op.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

template<typename T>
__global__ void FillingKernel(
    tensorflow::CudaLaunchConfig config,
    InputIndexer::FixedRankState input_indexer_state,
    const T* input, T* output) {
  // Local state for this GPU thread.
  InputIndexer::FixedRankState local_state = input_indexer_state;
  InputIndexer input_indexer(&local_state);

  CUDA_1D_KERNEL_LOOP(output_index, config.virtual_thread_count) {
    input_indexer.MoveToOutputIndex(output_index);
    output[output_index] = tensorflow::ldg(input + input_indexer.current_input_index());
  }
}

template<typename InputDataT>
void FillPeriodicTensorImpl<GPUDevice, InputDataT>::operator()(
      int rank,
      int adjustable_dimension,
      const std::vector<tensorflow::int64>& target_dimensions,
      const std::vector<tensorflow::int64>& dimension_ceiling,
      const std::vector<tensorflow::int64>& cumulative_dimensions,
      const std::vector<tensorflow::int64>& original_dimensions,
      tensorflow::OpKernelContext* context,
      const InputDataT* input,
      InputDataT* output,
      tensorflow::int64 output_size) {

    // TODO!! Provide Fallback to CPU implementation if rank too big!!.

  InputIndexer::FixedRankState indexer_state(
      rank,
      adjustable_dimension,
      target_dimensions,
      dimension_ceiling,
      cumulative_dimensions);
  indexer_state.InitializeIndexFactors(original_dimensions);
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  tensorflow::CudaLaunchConfig config = tensorflow::GetCudaLaunchConfig(
      output_size, d);

  FillingKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config, indexer_state, input, output);
}

template<typename T>
__global__ void FillingGradKernel(
    tensorflow::CudaLaunchConfig config,
    InputIndexer::FixedRankState input_indexer_state,
    const T* input_grad_data, T* output) {
  // Local state for this GPU thread.
  InputIndexer::FixedRankState local_state = input_indexer_state;
  InputIndexer input_indexer(&local_state);

  CUDA_1D_KERNEL_LOOP(input_grad_index, config.virtual_thread_count) {
    input_indexer.MoveToOutputIndex(input_grad_index);
    output[input_indexer.current_input_index()] =
          tensorflow::ldg(input_grad_data + input_grad_index);
  }
}

template<typename InputDataT>
void FillGradTensorImpl<GPUDevice, InputDataT>::operator()(
      int rank,
      int adjustable_dimension,
      const std::vector<tensorflow::int64>& target_dimensions,
      const std::vector<tensorflow::int64>& dimension_ceiling,
      const std::vector<tensorflow::int64>& cumulative_dimensions,
      const std::vector<tensorflow::int64>& original_dimensions,
      tensorflow::OpKernelContext* context,
      const InputDataT* input_grad_data,
      InputDataT* output,
      tensorflow::int64 new_size) {
    // TODO!! Provide Fallback to CPU implementation if rank too big!!.

  InputIndexer::FixedRankState indexer_state(
      rank,
      adjustable_dimension,
      target_dimensions,
      dimension_ceiling,
      cumulative_dimensions);
  indexer_state.InitializeIndexFactors(original_dimensions);
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  tensorflow::CudaLaunchConfig config = tensorflow::GetCudaLaunchConfig(
      new_size, d);

  FillingGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config, indexer_state, input_grad_data, output);
}

template class FillPeriodicTensorImpl<GPUDevice, float>;
template class FillPeriodicTensorImpl<GPUDevice, double>;
template class FillPeriodicTensorImpl<GPUDevice, tensorflow::int32>;
template class FillPeriodicTensorImpl<GPUDevice, tensorflow::int64>;


template class FillGradTensorImpl<GPUDevice, float>;
template class FillGradTensorImpl<GPUDevice, double>;
template class FillGradTensorImpl<GPUDevice, tensorflow::int32>;
template class FillGradTensorImpl<GPUDevice, tensorflow::int64>;

#endif
