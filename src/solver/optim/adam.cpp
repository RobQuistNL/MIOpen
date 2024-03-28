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

#include <miopen/optim/adam/solvers.hpp>

#include <miopen/adam.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/optim/adam/invoke_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace adam {

bool IsImprovementOverROCm([[maybe_unused]] const miopen::adam::ProblemDescription& problem)
{
    return true;
}

bool Adam::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                        const miopen::adam::ProblemDescription& problem) const
{
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

inline size_t AlignUp(size_t num, size_t align) { return (num + align - 1) / align * align; }

ConvSolution Adam::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::adam::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto param_dtype = miopen::GetDataType(problem.GetParamDesc().GetType());
        auto grad_dtype  = miopen::GetDataType(problem.GetGradDesc().GetType());

        const auto build_params = KernelBuildParameters{
            {"PARAM_TYPE", param_dtype == "bfloat16" ? "ushort" : param_dtype},
            {"GRAD_TYPE", grad_dtype == "bfloat16" ? "ushort" : grad_dtype},
        };

        auto kernel = KernelInfo{};

        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(1);

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.kernel_file = "MIOpenAdam.cpp";
        kernel.kernel_name = problem.IsAmp() ? "AmpAdamPacked" : "AdamPacked";

        result.construction_params.push_back(kernel);

        if(problem.IsAmp() && problem.ExistStepOut())
        {
            auto kernel_update_step        = kernel;
            kernel_update_step.kernel_name = "AdamUpdateStep";

            result.construction_params.push_back(kernel_update_step);
        }
    }

    constexpr size_t local_size = 512;
    if(problem.IsAmp())
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel_adam = handle_.Run(kernels[0]);
                decltype(auto) kernel_step = handle_.Run(kernels[1]);
                decltype(auto) params      = raw_params.CastTo<miopen::adam::InvokeParams>();
                decltype(auto) numel       = params.paramDesc->GetElementSize();
                auto elapsed               = 0.f;

                kernel_adam.ldims = {local_size, 1, 1};
                kernel_adam.gdims = {AlignUp(numel, local_size), 1, 1};

                kernel_adam(params.paramIn,
                            params.gradIn,
                            params.expAvgIn,
                            params.expAvgSqIn,
                            params.maxExpAvgSqIn,
                            params.gradScale,
                            params.foundInf,
                            params.stepIn,
                            params.lr,
                            params.beta1,
                            params.beta2,
                            params.weight_decay,
                            params.eps,
                            params.amsgrad,
                            params.maximize,
                            params.paramOut,
                            nullptr,
                            params.expAvgOut,
                            params.expAvgSqOut,
                            params.maxExpAvgSqOut,
                            params.stepOut,
                            numel);

                if(params.stepOut != nullptr)
                {
                    if(handle_.IsProfilingEnabled())
                        elapsed = handle_.GetKernelTime();

                    kernel_step(params.foundInf, params.stepOut);

                    if(handle_.IsProfilingEnabled())
                    {
                        elapsed += handle_.GetKernelTime();
                        handle_.ResetKernelTime();
                        handle_.AccumKernelTime(elapsed);
                    };
                }
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::adam::InvokeParams>();
                decltype(auto) numel  = params.paramDesc->GetElementSize();

                kernel.ldims = {local_size, 1, 1};
                kernel.gdims = {AlignUp(numel, local_size), 1, 1};

                kernel(params.paramIn,
                       params.gradIn,
                       params.expAvgIn,
                       params.expAvgSqIn,
                       params.maxExpAvgSqIn,
                       params.step,
                       params.lr,
                       params.beta1,
                       params.beta2,
                       params.weight_decay,
                       params.eps,
                       params.amsgrad,
                       params.maximize,
                       params.paramOut,
                       params.expAvgOut,
                       params.expAvgSqOut,
                       params.maxExpAvgSqOut,
                       numel);
            };
        };
    }

    return result;
}

} // namespace adam

} // namespace solver

} // namespace miopen
