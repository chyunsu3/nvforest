/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/cuda_stream.hpp>
#include <nvforest/device_type.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <cuda/stream>
#include <thrust/device_vector.h>

#include <gtest/gtest.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nvforest {

std::unique_ptr<treelite::Model> get_tree_model()
{
  auto model_builder = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat32,
    treelite::TypeInfo::kFloat32,
    treelite::model_builder::Metadata{
      1,
      treelite::TaskType::kRegressor,
      false,
      1,
      {1},
      {1, 1},
    },
    treelite::model_builder::TreeAnnotation{2, {0, 0}, {0, 0}},
    treelite::model_builder::PostProcessorFunc{"identity"},
    std::vector<double>{0.0});
  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kLT, 1, 2);
  model_builder->EndNode();
  model_builder->StartNode(1);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();
  model_builder->StartNode(2);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();
  model_builder->EndTree();

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kLT, 1, 2);
  model_builder->EndNode();
  model_builder->StartNode(1);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();
  model_builder->StartNode(2);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();
  model_builder->EndTree();

  return model_builder->CommitModel();
}

std::vector<float> repeat(std::vector<float> const& x, std::size_t n_repeat)
{
  std::vector<float> ret(x.size() * n_repeat);
  for (std::size_t i = 0; i < n_repeat; ++i) {
    for (std::size_t j = 0; j < x.size(); j++) {
      ret[i * x.size() + j] = x[j];
    }
  }
  return ret;
}

template <typename BufferType>
class GPUModelInferenceTest : public testing::Test {};

using BufferTypes = testing::Types<std::vector<float>, thrust::device_vector<float>>;

struct BufferTypeNames {
  template <typename BufferType>
  static std::string GetName(int)
  {
    if constexpr (std::is_same_v<BufferType, std::vector<float>>) {
      return "CPU";
    } else {
      return "GPU";
    }
  }
};

TYPED_TEST_SUITE(GPUModelInferenceTest, BufferTypes, BufferTypeNames);

TYPED_TEST(GPUModelInferenceTest, Predict)
{
  constexpr auto mem_type =
    std::is_same_v<TypeParam, std::vector<float>> ? device_type::cpu : device_type::gpu;
  auto tree_model     = get_tree_model();
  auto nvforest_model = import_from_treelite_model(
    *tree_model, tree_layout::breadth_first, index_type{}, false, device_type::gpu);

  // Test with a single stream
  {
    auto input       = TypeParam{-1.0f, 1.0f};
    auto const cases = std::array{
      std::pair{infer_kind::default_kind, TypeParam{-2.0f, 2.0f}},
      std::pair{infer_kind::per_tree, TypeParam{-1.0f, -1.0f, 1.0f, 1.0f}},
      std::pair{infer_kind::leaf_id, TypeParam{1.0f, 1.0f, 2.0f, 2.0f}},
    };

    cuda::device_ref device{0};
    cuda::stream stream{device};
    for (auto const& [infer_type, expected] : cases) {
      SCOPED_TRACE(static_cast<int>(infer_type));
      auto output = TypeParam(expected.size());
      nvforest_model.predict(stream.get(),
                             thrust::raw_pointer_cast(output.data()),
                             thrust::raw_pointer_cast(input.data()),
                             input.size(),
                             mem_type,
                             mem_type,
                             infer_type);
      stream.sync();
      EXPECT_EQ(output, expected);
    }
  }

  // Test with multiple streams
  {
    std::size_t n_repeat = 256;
    auto input           = TypeParam(repeat(std::vector<float>{-1.0f, 1.0f}, n_repeat));
    auto const cases     = std::array{
      std::pair{infer_kind::default_kind,
                TypeParam(repeat(std::vector<float>{-2.0f, 2.0f}, n_repeat))},
      std::pair{infer_kind::per_tree,
                TypeParam(repeat(std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f}, n_repeat))},
      std::pair{infer_kind::leaf_id,
                TypeParam(repeat(std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, n_repeat))},
    };

    cuda::device_ref device{0};
    std::size_t n_streams         = 4;
    std::size_t current_stream_id = 0;
    std::vector<cuda::stream> stream_pool;
    for (std::size_t i = 0; i < n_streams; ++i) {
      stream_pool.emplace_back(device);
    }
    auto stream_pool_next = [&]() {
      return nvforest::cuda_stream{stream_pool[current_stream_id++].get()};
    };
    auto stream_pool_size = [&]() { return stream_pool.size(); };
    for (auto const& [infer_type, expected] : cases) {
      SCOPED_TRACE(static_cast<int>(infer_type));
      auto output = TypeParam(expected.size());
      nvforest_model.predict(stream_pool_next,
                             stream_pool_size,
                             thrust::raw_pointer_cast(output.data()),
                             thrust::raw_pointer_cast(input.data()),
                             input.size(),
                             mem_type,
                             mem_type,
                             infer_type);
      for (std::size_t i = 0; i < n_streams; ++i) {
        stream_pool[i].sync();
      }
      EXPECT_EQ(output, expected);
    }
  }
}

// TODO(chyunsu3): Remove once we remove the RAFT dependency in 27.02.
TEST(Inference, GPUModelWithRAFTHandle)
{
  auto tree_model     = get_tree_model();
  auto nvforest_model = import_from_treelite_model(
    *tree_model, tree_layout::breadth_first, index_type{}, false, device_type::gpu);

  std::size_t n_repeat = 256;
  auto input = thrust::device_vector<float>(repeat(std::vector<float>{-1.0f, 1.0f}, n_repeat));
  auto const cases = std::array{
    std::pair{infer_kind::default_kind,
              thrust::device_vector<float>(repeat(std::vector<float>{-2.0f, 2.0f}, n_repeat))},
    std::pair{
      infer_kind::per_tree,
      thrust::device_vector<float>(repeat(std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f}, n_repeat))},
    std::pair{
      infer_kind::leaf_id,
      thrust::device_vector<float>(repeat(std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, n_repeat))},
  };

  std::size_t n_streams = 4;
  raft::handle_t handle{cuda::stream_ref{cudaStreamPerThread},
                        std::make_shared<rmm::cuda_stream_pool>(n_streams)};
  for (auto const& [infer_type, expected] : cases) {
    SCOPED_TRACE(static_cast<int>(infer_type));
    auto output = thrust::device_vector<float>(expected.size());
    nvforest_model.predict([&] { return cuda_stream{handle.get_next_usable_stream().get()}; },
                           [&] { return handle.get_stream_pool_size(); },
                           thrust::raw_pointer_cast(output.data()),
                           thrust::raw_pointer_cast(input.data()),
                           input.size(),
                           device_type::gpu,
                           device_type::gpu,
                           infer_type);
    handle.sync_stream_pool();
    EXPECT_EQ(output, expected);
  }
}

}  // namespace nvforest
