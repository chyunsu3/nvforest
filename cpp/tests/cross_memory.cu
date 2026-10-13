/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/cuda_stream.hpp>
#include <nvforest/device_type.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>

#include <gtest/gtest.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <array>
#include <utility>
#include <vector>

namespace nvforest {

TEST(CrossMemory, GpuModelWithCpuInputAndOutput)
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

  auto tree_model     = model_builder->CommitModel();
  auto nvforest_model = import_from_treelite_model(
    *tree_model, tree_layout::breadth_first, index_type{}, false, device_type::gpu);
  auto input       = std::vector<float>{-1.0f, 1.0f};
  auto const cases = std::array{
    std::pair{infer_kind::default_kind, std::vector<float>{-2.0f, 2.0f}},
    std::pair{infer_kind::per_tree, std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f}},
    std::pair{infer_kind::leaf_id, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}},
  };

  for (auto const& [infer_type, expected] : cases) {
    SCOPED_TRACE(static_cast<int>(infer_type));
    auto output = std::vector<float>(expected.size());
    nvforest_model.predict(cuda_stream{},
                           output.data(),
                           input.data(),
                           input.size(),
                           device_type::cpu,
                           device_type::cpu,
                           infer_type);
    EXPECT_EQ(output, expected);
  }
}

}  // namespace nvforest
