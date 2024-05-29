/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/embedding/solvers.hpp>

#include <miopen/datatype.hpp>
#include <miopen/embedding.hpp>
#include <miopen/embedding/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor_view_utils.hpp>

namespace miopen {

namespace solver {

namespace embedding {

bool EmbeddingBackward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::embedding::ProblemDescription& problem) const
{
    if(problem.IsForward())
        return false;
    /*
        if(!problem.IsAllContigous())
            return false;
    */
    return true;
}

constexpr uint64_t DivCeil(uint64_t numer, uint64_t denom) { return (numer + denom - 1) / denom; }

ConvSolution
EmbeddingBackward::GetSolution(const ExecutionContext& context,
                               const miopen::embedding::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        constexpr size_t local_size = 256;
        constexpr int alpha         = 64;
        auto dtype                  = problem.GetOutputDesc().GetType();

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"LOCAL_SIZE", local_size},
            {"ALPHA", alpha},
        };

        const auto embedding_dim  = problem.GetEmbeddingDim();
        const auto num_embeddings = problem.GetNumEmbeddings();
        const auto input_size     = problem.GetInputDesc().GetElementSize();
        size_t grid_size          = 1;

        auto kernel = KernelInfo{};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
        kernel.kernel_file  = "MIOpenEmbedding.cpp";

        if(problem.IsDeterministicMode())
        {
            kernel.kernel_name = "EmbeddingBackwardContiguous";
            grid_size          = DivCeil(embedding_dim, local_size);
        }
        else if(num_embeddings <= 32)
        {
            kernel.kernel_name = "EmbeddingBackwardSmallNumEmbeddingsTraverseContiguous";
            grid_size          = DivCeil(alpha * num_embeddings * embedding_dim, local_size);
        }
        else
        {
            kernel.kernel_name = "EmbeddingBackwardContiguousAtomic";
            grid_size          = DivCeil(input_size * embedding_dim, local_size);
        }

        kernel.l_wk.push_back(local_size);
        kernel.g_wk.push_back(grid_size);

        result.construction_params.push_back(kernel);
    }

    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::embedding::InvokeParams>();
                auto elapsed          = 0.f;

                size_t num_embeddings = params.weightGradDesc->GetLengths().front();
                auto embedding_dim    = params.outputGradDesc->GetLengths().back();
                auto input_size       = params.inputDesc->GetElementSize();

                float zero = 0.0f;
                SetTensor(handle_, *params.weightGradDesc, params.weight_grad, &zero);

                if(handle_.IsProfilingEnabled())
                    elapsed = handle_.GetKernelTime();

                kernel(params.input,
                       params.output_grad,
                       params.weight_grad,
                       params.indices_freq,
                       params.error,
                       embedding_dim,
                       input_size,
                       num_embeddings,
                       params.padding_idx);

                if(handle_.IsProfilingEnabled())
                {
                    elapsed += handle_.GetKernelTime();
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                }
            };
        };
    }

    return result;
}

} // Namespace embedding

} // namespace solver

} // namespace miopen
