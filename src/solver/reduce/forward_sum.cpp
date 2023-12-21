/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/reduce/invoke_params.hpp>
#include <miopen/reduce/solvers.hpp>
#include <miopen/sum.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace reduce {

size_t get_reqd_work_item_cnt(const ExecutionContext& context)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * context.GetStream().GetMaxComputeUnits() * 4);
}

size_t get_reqd_work_item_cnt(const Handle& handle)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * handle.GetMaxComputeUnits() * 4);
}

size_t get_parallelism_size(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    size_t parallelism_size = 1ULL;
    while(parallelism_size * output_numel < reqd_work_item_cnt &&
          parallelism_size < std::sqrt(reduce_size))
    {
        parallelism_size *= 2ULL;
    }
    return parallelism_size;
}

bool is_parallelism(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    return !(output_numel > reqd_work_item_cnt) &&
           (output_numel * reduce_size > reqd_work_item_cnt);
}

bool IsImprovementOverROCm(const ExecutionContext& context,
                           const miopen::reduce::ProblemDescription& problem)
{
    auto xdims     = problem.GetXDesc().GetLengths();
    auto dims      = problem.GetDims();
    auto dims_size = problem.GetDims_size();
    auto sort_dims = dims;

    std::sort(sort_dims, sort_dims + (dims_size - 1));

    auto output_dims = xdims;

    for(int32_t idx = 0; idx < dims_size; idx++)
    {
        auto dim         = sort_dims[dims_size - 1 - idx];
        output_dims[dim] = 1;

        auto reduce_size  = xdims[dim];
        auto output_numel = std::accumulate(
            output_dims.begin(), output_dims.end(), 1ULL, std::multiplies<size_t>());

        auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

        if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
        {
            // It's large enough to parallelization, but calling the kernel twice is overhead.
            // For cases smaller than this, ROCm pytorch performed better.
            bool is_improvement_ROCm = (output_numel * reduce_size < reqd_work_item_cnt * 64);
            // But the reduce size is small, MIOpen HIP performed better.
            bool is_reduce_large = (reduce_size > 64);

            if(is_improvement_ROCm && is_reduce_large)
                return false;
        }
    }
    return true;
}

bool SumForward::IsApplicable(const ExecutionContext& context,
                              const miopen::reduce::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightDim())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsNotLastDim())
        return false;
    if(!IsImprovementOverROCm(context, problem))
        return false;
    return true;
}

ConvSolution SumForward::GetSolution(const ExecutionContext& context,
                                     const miopen::reduce::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype     = problem.GetXDesc().GetType();
    auto xdims     = problem.GetXDesc().GetLengths();
    auto dims      = problem.GetDims();
    auto dims_size = problem.GetDims_size();
    auto sort_dims = dims;

    std::sort(sort_dims, sort_dims + (dims_size - 1));

    auto output_dims = xdims;

    for(int32_t idx = 0; idx < dims_size; idx++)
    {
        auto dim         = sort_dims[dims_size - 1 - idx];
        output_dims[dim] = 1;

        auto reduce_size  = xdims[dim];
        auto output_numel = std::accumulate(
            output_dims.begin(), output_dims.end(), 1ULL, std::multiplies<size_t>());

        auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

        if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
        {
            auto parallelism_size =
                get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);

            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(parallelism_size * output_numel, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenSum.cpp";
            kernel.kernel_name = "SumParallelFwdContiguous";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            };

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
        }

        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(output_numel, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenSum.cpp";
            kernel.kernel_name = "SumFwdContiguous";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            };

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
        }
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::reduce::InvokeParams>();

            int32_t kernels_idx = 0;
            auto xdims          = params.xDesc->GetLengths();
            auto dims           = params.dims;
            auto dims_size      = params.dims_size;
            auto sort_dims      = dims;

            std::sort(sort_dims, sort_dims + (dims_size - 1));

            std::size_t x_workspace_offset = 0;
            std::size_t y_workspace_offset = 0;
            auto output_dims               = xdims;

            auto elapsed = 0.f;

            for(int32_t idx = 0; idx < dims_size; idx++)
            {
                auto dim = sort_dims[dims_size - 1 - idx];

                auto inner_size = 1ULL;
                for(int32_t i = dim + 1; i < output_dims.size(); i++)
                {
                    inner_size *= output_dims[i];
                }

                output_dims[dim] = 1;

                auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_);

                auto reduce_size  = xdims[dim];
                auto output_numel = std::accumulate(
                    output_dims.begin(), output_dims.end(), 1ULL, std::multiplies<size_t>());

                if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
                {
                    decltype(auto) parallel_kernel = handle_.Run(kernels[kernels_idx++]);
                    decltype(auto) kernel          = handle_.Run(kernels[kernels_idx++]);

                    auto parallelism_size =
                        get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);

                    parallel_kernel(idx == 0 ? params.x : params.workspace,
                                    params.workspace,
                                    output_numel,
                                    reduce_size,
                                    parallelism_size,
                                    inner_size,
                                    static_cast<bool>(params.nanPropagation),
                                    x_workspace_offset,
                                    y_workspace_offset);

                    if(handle_.IsProfilingEnabled())
                    {
                        elapsed += handle_.GetKernelTime();
                        handle_.ResetKernelTime();
                        handle_.AccumKernelTime(elapsed);
                    };

                    x_workspace_offset = y_workspace_offset;
                    y_workspace_offset = y_workspace_offset + parallelism_size * output_numel;

                    kernel(params.workspace,
                           idx == (dims_size - 1) ? params.y : params.workspace,
                           output_numel,
                           parallelism_size,
                           inner_size,
                           dim,
                           static_cast<bool>(params.nanPropagation),
                           x_workspace_offset,
                           idx == (dims_size - 1) ? 0 : y_workspace_offset);

                    if(handle_.IsProfilingEnabled())
                    {
                        elapsed += handle_.GetKernelTime();
                        handle_.ResetKernelTime();
                        handle_.AccumKernelTime(elapsed);
                    };

                    x_workspace_offset = y_workspace_offset;
                    y_workspace_offset = y_workspace_offset + output_numel;
                }
                else
                {
                    decltype(auto) kernel = handle_.Run(kernels[kernels_idx++]);

                    kernel(idx == 0 ? params.x : params.workspace,
                           idx == (dims_size - 1) ? params.y : params.workspace,
                           output_numel,
                           reduce_size,
                           inner_size,
                           dim,
                           static_cast<bool>(params.nanPropagation),
                           x_workspace_offset,
                           idx == (dims_size - 1) ? 0 : y_workspace_offset);

                    if(handle_.IsProfilingEnabled())
                    {
                        elapsed += handle_.GetKernelTime();
                        handle_.ResetKernelTime();
                        handle_.AccumKernelTime(elapsed);
                    };

                    x_workspace_offset = y_workspace_offset;
                    y_workspace_offset = y_workspace_offset + output_numel;
                }
            }
        };
    };

    return result;
}

std::size_t SumForward::GetWorkspaceSize(const ExecutionContext& context,
                                         const miopen::reduce::ProblemDescription& problem) const
{
    auto xdims     = problem.GetXDesc().GetLengths();
    auto dims      = problem.GetDims();
    auto dims_size = problem.GetDims_size();
    auto sort_dims = dims;

    std::sort(sort_dims, sort_dims + (dims_size - 1));

    std::size_t workspacesize = 0;
    auto output_dims          = xdims;

    for(int32_t idx = 0; idx < dims_size; idx++)
    {
        auto dim         = sort_dims[dims_size - 1 - idx];
        output_dims[dim] = 1;

        auto reduce_size  = xdims[dim];
        auto output_numel = std::accumulate(
            output_dims.begin(), output_dims.end(), 1ULL, std::multiplies<size_t>());

        auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

        if(idx != dims_size - 1)
        {
            workspacesize += output_numel * get_data_size(problem.GetXDesc().GetType());
        }

        if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
        {
            auto parallelism_size =
                get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);

            workspacesize +=
                parallelism_size * output_numel * get_data_size(problem.GetXDesc().GetType());
        }
    }

    return workspacesize;
}

} // namespace reduce

} // namespace solver

} // namespace miopen
