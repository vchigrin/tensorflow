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

#ifndef TENSORFLOW_KERNELS_INPUT_INDEXER_H_
#define TENSORFLOW_KERNELS_INPUT_INDEXER_H_

#include <vector>
#include <string.h>

#include "tensorflow/core/platform/types.h"

#if defined(__CUDACC__)
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class InputIndexer {
 public:
  struct StateBase {
    const int adjustable_dimension_;
    const int rank_;
    tensorflow::int64 current_input_index_;
    tensorflow::int64 adjustable_dimension_carriage_sum_;

    const tensorflow::int64* cumulative_dimensions_;
    tensorflow::int64* index_factors_;
    const tensorflow::int64* dimension_ceiling_;
    const tensorflow::int64* target_dimensions_;
    tensorflow::int64* output_indices_;
    tensorflow::int64* clamped_output_indices_;

    StateBase(int adjustable_dimension,
              int rank,
              const tensorflow::int64* cumulative_dimensions,
              tensorflow::int64* index_factors,
              const tensorflow::int64* dimension_ceiling,
              const tensorflow::int64* target_dimensions,
              tensorflow::int64* output_indices,
              tensorflow::int64* clamped_output_indices)
        : adjustable_dimension_(adjustable_dimension),
          rank_(rank),
          current_input_index_(0),
          adjustable_dimension_carriage_sum_(0),
          cumulative_dimensions_(cumulative_dimensions),
          index_factors_(index_factors),
          dimension_ceiling_(dimension_ceiling),
          target_dimensions_(target_dimensions),
          output_indices_(output_indices),
          clamped_output_indices_(clamped_output_indices) {}
    void InitializeIndexFactors(
        const std::vector<tensorflow::int64>& original_dimensions) {
      tensorflow::int64 last_index_factor = 1;
      for (auto r = rank_ - 1; r >= 0; --r) {
        index_factors_[r] = last_index_factor;
        last_index_factor *= original_dimensions[r];
      }
    }
  };

  struct VariableRankState : public StateBase {
    VariableRankState(
        int rank,
        int adjustable_dimension,
        const std::vector<tensorflow::int64>& target_dimensions,
        const std::vector<tensorflow::int64>& dimension_ceiling,
        const std::vector<tensorflow::int64>& cumulative_dimensions)
      : StateBase(adjustable_dimension,
                  rank,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr),
        target_dimensions_holder_(target_dimensions),
        dimension_ceiling_holder_(dimension_ceiling),
        cumulative_dimensions_holder_(cumulative_dimensions) {

      target_dimensions_ = target_dimensions_holder_.data();
      dimension_ceiling_ = dimension_ceiling_holder_.data();
      cumulative_dimensions_ = cumulative_dimensions_holder_.data();

      index_factors_holder_.resize(rank);
      index_factors_ = index_factors_holder_.data();

      output_indices_holder_.resize(rank);
      output_indices_ = output_indices_holder_.data();

      clamped_output_indices_holder_.resize(rank);
      clamped_output_indices_ = clamped_output_indices_holder_.data();
    }

    const std::vector<tensorflow::int64> target_dimensions_holder_;
    const std::vector<tensorflow::int64> dimension_ceiling_holder_;
    const std::vector<tensorflow::int64> cumulative_dimensions_holder_;

    std::vector<tensorflow::int64> index_factors_holder_;
    std::vector<tensorflow::int64> output_indices_holder_;
    std::vector<tensorflow::int64> clamped_output_indices_holder_;
  };

  static const int kMaxSupportedFixedRank = 4;

  struct FixedRankState : public StateBase {
    FixedRankState(
        int rank,
        int adjustable_dimension,
        const std::vector<tensorflow::int64>& target_dimensions,
        const std::vector<tensorflow::int64>& dimension_ceiling,
        const std::vector<tensorflow::int64>& cumulative_dimensions)
      : StateBase(adjustable_dimension,
                  rank,
                  cumulative_dimensions_holder_,
                  index_factors_holder_,
                  dimension_ceiling_holder_,
                  target_dimensions_holder_,
                  output_indices_holder_,
                  clamped_output_indices_holder_) {
     // DCHECK_LE(rank, kMaxSupportedFixedRank);

      std::copy(
          target_dimensions.begin(),
          target_dimensions.end(),
          target_dimensions_holder_);
      std::copy(
          dimension_ceiling.begin(),
          dimension_ceiling.end(), dimension_ceiling_holder_);
      std::copy(
          cumulative_dimensions.begin(),
          cumulative_dimensions.end(), cumulative_dimensions_holder_);
    }

    const FixedRankState& operator=(const FixedRankState&) = delete;

    CUDA_CALLABLE FixedRankState(const FixedRankState& second)
        : StateBase(second) {
      memcpy(this, &second, sizeof(second));

      cumulative_dimensions_ = cumulative_dimensions_holder_;
      index_factors_ = index_factors_holder_;
      dimension_ceiling_ = dimension_ceiling_holder_;
      target_dimensions_ = target_dimensions_holder_;
      output_indices_ = output_indices_holder_;
      clamped_output_indices_ = clamped_output_indices_holder_;
    }

    tensorflow::int64 target_dimensions_holder_[kMaxSupportedFixedRank];
    tensorflow::int64 dimension_ceiling_holder_[kMaxSupportedFixedRank];
    tensorflow::int64 cumulative_dimensions_holder_[kMaxSupportedFixedRank];

    tensorflow::int64 index_factors_holder_[kMaxSupportedFixedRank];
    tensorflow::int64 output_indices_holder_[kMaxSupportedFixedRank];
    tensorflow::int64 clamped_output_indices_holder_[kMaxSupportedFixedRank];
  };



  CUDA_CALLABLE InputIndexer(StateBase* state)
    : state_(state) {}

  CUDA_CALLABLE tensorflow::int64 current_input_index() const {
    return state_->current_input_index_;
  }

  CUDA_CALLABLE void MoveToOutputIndex(tensorflow::int64 output_index) {
    // un-rasterize the output index
    auto last_reduced_i = output_index;
    for (auto r = state_->rank_ - 1; r >= 0; --r) {
      state_->output_indices_[r] = last_reduced_i % state_->target_dimensions_[r];
      last_reduced_i =
          (last_reduced_i - state_->output_indices_[r]) / state_->target_dimensions_[r];
    }

    tensorflow::int64 carriage_sum = 0;
    for (int qi = 0; qi < state_->rank_; ++qi) {
      if (qi == state_->adjustable_dimension_)
        continue;
      carriage_sum += state_->cumulative_dimensions_[qi] *
               (state_->output_indices_[qi] % state_->dimension_ceiling_[qi]);
    }
    state_->adjustable_dimension_carriage_sum_ = carriage_sum;

    // rasterize the input index
    for (auto r = state_->rank_ - 1; r >= 0; --r) {
      if (r != state_->adjustable_dimension_)
        state_->clamped_output_indices_[r] = state_->output_indices_[r] / state_->dimension_ceiling_[r];
      else {
        RecomputeClampedAdjustableDimensionIndex();
      }
    }
    state_->current_input_index_ = 0;
    for (auto r = state_->rank_ - 1; r >= 0; --r) {
      state_->current_input_index_ += state_->index_factors_[r] * state_->clamped_output_indices_[r];
    }
  }

  void IncrementOutputIndex() {
    // un-rasterize the output index
    for (auto r = state_->rank_ - 1; r >= 0; --r) {
      auto old_carriage_sum_increment = state_->cumulative_dimensions_[r] * (state_->output_indices_[r] % state_->dimension_ceiling_[r]);
      state_->output_indices_[r] = (state_->output_indices_[r] + 1) % state_->target_dimensions_[r];
      if (r != state_->adjustable_dimension_) {
        auto new_output_index = state_->output_indices_[r] / state_->dimension_ceiling_[r];
        state_->current_input_index_ += (new_output_index - state_->clamped_output_indices_[r]) * state_->index_factors_[r];

        state_->clamped_output_indices_[r] = new_output_index;

        auto new_carriage_sum_increment = state_->cumulative_dimensions_[r] * (state_->output_indices_[r] % state_->dimension_ceiling_[r]);

        state_->adjustable_dimension_carriage_sum_ = state_->adjustable_dimension_carriage_sum_ - old_carriage_sum_increment + new_carriage_sum_increment;
      }

      if (state_->output_indices_[r] != 0) {
        // No more carries to higher indices.
        break;
      }
    }
    auto old_clamped_output_index = state_->clamped_output_indices_[state_->adjustable_dimension_];
    RecomputeClampedAdjustableDimensionIndex();
    state_->current_input_index_ += (state_->clamped_output_indices_[state_->adjustable_dimension_] - old_clamped_output_index) * state_->index_factors_[state_->adjustable_dimension_];
  }

 private:
  CUDA_CALLABLE void RecomputeClampedAdjustableDimensionIndex() {
    tensorflow::int64 index = state_->adjustable_dimension_carriage_sum_;
    index *= state_->target_dimensions_[state_->adjustable_dimension_];
    index += state_->output_indices_[state_->adjustable_dimension_];
    state_->clamped_output_indices_[state_->adjustable_dimension_] = index;
  }
  StateBase* state_;
};


#endif  // TENSORFLOW_KERNELS_INPUT_INDEXER_H_
