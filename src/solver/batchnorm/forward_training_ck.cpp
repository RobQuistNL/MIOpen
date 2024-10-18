
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

#include <miopen/batchnorm/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_BN_FWD_TRAINING)

namespace miopen {
namespace solver {
namespace batchnorm {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;
using index_t       = int32_t;

constexpr index_t Rank                  = 4;
constexpr index_t NumBatchNormReduceDim = 3;

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
using DeviceOpBNFwdTrainingPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        ck::tensor_operation::device::DeviceBatchNormFwd<XDataType,
                                                         YDataType,
                                                         AccDataType,
                                                         ScaleDataType,
                                                         BiasDataType,
                                                         MeanVarDataType,
                                                         PassThroughOp,
                                                         Rank,
                                                         NumBatchNormReduceDim>>;

struct CKArgsBNormFwdTraining
{
    CKArgsBNormFwdTraining(const miopen::batchnorm::ProblemDescription& problem)
    {
        std::copy(problem.GetXDesc().GetLengths().begin(),
                  problem.GetXDesc().GetLengths().end(),
                  xyLengths.begin());

        std::copy(problem.GetXDesc().GetStrides().begin(),
                  problem.GetXDesc().GetStrides().end(),
                  xyStrides.begin());
        arrScaleBiasMeanVarLengths[0] = xyLengths[1]; // get channel
        arrScaleBiasMeanVarStrides[0] = 1;

        // prep for CK
        std::sort(xyStrides.begin(), xyStrides.end(), std::greater<>());
        std::rotate(xyLengths.begin() + 1, xyLengths.begin() + 2, xyLengths.end());
    }

    CKArgsBNormFwdTraining(const CKArgsBNormFwdTraining&) = default;
    CKArgsBNormFwdTraining(CKArgsBNormFwdTraining&&)      = default;
    CKArgsBNormFwdTraining& operator=(const CKArgsBNormFwdTraining&) = default;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& data_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(xyLengths,
                                                xyStrides,
                                                xyStrides,
                                                reduceDims,
                                                arrScaleBiasMeanVarLengths,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                arrScaleBiasMeanVarStrides,
                                                data_ctx.x,
                                                data_ctx.bnScale,
                                                data_ctx.bnBias,
                                                data_ctx.epsilon,
                                                PassThroughOp{},
                                                data_ctx.y,
                                                data_ctx.resultSaveMean,
                                                data_ctx.resultSaveInvVariance,
                                                data_ctx.expAvgFactor,
                                                data_ctx.resultRunningMean,
                                                data_ctx.resultRunningVariance);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& invoker_ptr) const
    {
        auto arg_ptr = MakeArgPtr(invoker_ptr, miopen::batchnorm::FwdTrainInvokeParams{});
        return invoker_ptr->IsSupportedArgument(arg_ptr.get());
    }

    std::array<ck::index_t, Rank> xyLengths;
    std::array<ck::index_t, Rank> xyStrides;
    std::vector<int> invariantDims;

    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;

    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};
};

// template <typename XDataType,
//           typename YDataType,
//           typename AccDataType,
//           typename ScaleDataType,
//           typename BiasDataType,
//           typename MeanVarDataType>
// static bool CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
// {
//     return IsCKApplicable<DeviceOpBNFwdTrainingPtrs<XDataType,
//                                                     YDataType,
//                                                     AccDataType,
//                                                     ScaleDataType,
//                                                     BiasDataType,
//                                                     MeanVarDataType>,
//                           CKArgsBNormFwdTraining>(problem);
// }

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& bn_problem)
{
    const auto& valid_kernel_ids = FillValidKernelsIDs<DeviceOpBNFwdTrainingPtrs<XDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 ScaleDataType,
                                                                                 BiasDataType,
                                                                                 MeanVarDataType>,
                                                       CKArgsBNormFwdTraining>(bn_problem);
    assert(!valid_kernel_ids.empty());
    const auto& kernel_id = valid_kernel_ids[0];
    return InitAnyInvokerFactory<DeviceOpBNFwdTrainingPtrs<XDataType,
                                                           YDataType,
                                                           AccDataType,
                                                           ScaleDataType,
                                                           BiasDataType,
                                                           MeanVarDataType>,
                                 CKArgsBNormFwdTraining,
                                 miopen::batchnorm::FwdTrainInvokeParams>(bn_problem, kernel_id);
}

//====================== start

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
void PerformanceConfigBnCKFwdTraining::Init(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
    const auto& args       = CKArgsBNormFwdTraining{problem_desc};
    const auto bn_fwd_ptrs = DeviceOpBNFwdTrainingPtrs<XDataType,
                                                       YDataType,
                                                       AccDataType,
                                                       ScaleDataType,
                                                       BiasDataType,
                                                       MeanVarDataType>::GetInstances();
    assert(!bn_fwd_ptrs.empty());
    for(const auto& it : bn_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    0.0,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    0.0,
                                                    nullptr,
                                                    nullptr);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString());
        }
    }

    assert(!valid_kernels.empty());
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
bool PerformanceConfigBnCKFwdTraining::CheckIsSupportCKArgs(
    const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& args       = CKArgsBNormFwdTraining{problem};
    const auto bn_fwd_ptrs = DeviceOpBNFwdTrainingPtrs<XDataType,
                                                       YDataType,
                                                       AccDataType,
                                                       ScaleDataType,
                                                       BiasDataType,
                                                       MeanVarDataType>::GetInstances();
    assert(!bn_fwd_ptrs.empty());

    int i = 0;
    for(; i < bn_fwd_ptrs.size(); i++)
    {
        if(bn_fwd_ptrs[i]->GetTypeString() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr = bn_fwd_ptrs[i]->MakeArgumentPointer(args.xyLengths,
                                                            args.xyStrides,
                                                            args.xyStrides,
                                                            args.reduceDims,
                                                            args.arrScaleBiasMeanVarLengths,
                                                            args.arrScaleBiasMeanVarStrides,
                                                            args.arrScaleBiasMeanVarStrides,
                                                            args.arrScaleBiasMeanVarStrides,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            0.0,
                                                            PassThroughOp{},
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            0.0,
                                                            nullptr,
                                                            nullptr);
    return bn_fwd_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

void PerformanceConfigBnCKFwdTraining::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: Init<F16, F16, F32, F16, F16, F32>(problem_desc); break;
    case miopenBFloat16: Init<BF16, BF16, F32, BF16, BF16, F32>(problem_desc); break;
    case miopenFloat: Init<F32, F32, F32, F32, F32, F32>(problem_desc); break;
    case miopenDouble: Init<F64, F64, F64, F64, F64, F64>(problem_desc); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigBnCKFwdTraining::SetNextValue(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(problem_desc);
        assert(!valid_kernels.empty());
        return true;
    }
    if((this->index + 1) < valid_kernels.size())
    {
        ++this->index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
        return false;
#endif
}

bool PerformanceConfigBnCKFwdTraining::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigBnCKFwdTraining::IsValid(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem_desc;
    return false;
#else
    switch(problem_desc.GetXDesc().GetType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<F16, F16, F32, F16, F16, F32>(problem_desc);
    case miopenBFloat16:
        return CheckIsSupportCKArgs<BF16, BF16, F32, BF16, BF16, F32>(problem_desc);
    case miopenFloat: return CheckIsSupportCKArgs<F32, F32, F32, F32, F32, F32>(problem_desc);
    case miopenDouble: return CheckIsSupportCKArgs<F64, F64, F64, F64, F64, F64>(problem_desc);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigBnCKFwdTraining::operator==(
    const PerformanceConfigBnCKFwdTraining& other) const
{
    return this->kernel_id == other.kernel_id;
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
static int CheckCKApplicability(const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& args       = CKArgsBNormFwdTraining{problem};
    const auto bn_fwd_ptrs = DeviceOpBNFwdTrainingPtrs<XDataType,
                                                       YDataType,
                                                       AccDataType,
                                                       ScaleDataType,
                                                       BiasDataType,
                                                       MeanVarDataType>::GetInstances();
    assert(!bn_fwd_ptrs.empty());
    int count = 0;
    for(const auto& it : bn_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.xyStrides,
                                                    args.reduceDims,
                                                    args.arrScaleBiasMeanVarLengths,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    args.arrScaleBiasMeanVarStrides,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    0.0,
                                                    PassThroughOp{},
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    0.0,
                                                    nullptr,
                                                    nullptr);
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            return count;
        }
        count++;
    }
    return -1;
}

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
ConvSolution InvokerFactoryMakerNHWC(const miopen::batchnorm::ProblemDescription& bn_problem,
                                     const PerformanceConfigBnCKFwdTraining& config)
{
    ConvSolution result;
    auto bn_fwd_ptrs = DeviceOpBNFwdTrainingPtrs<XDataType,
                                                 YDataType,
                                                 AccDataType,
                                                 ScaleDataType,
                                                 BiasDataType,
                                                 MeanVarDataType>::GetInstances();

    assert(config.index >= 0 && !bn_fwd_ptrs.empty() && config.index < bn_fwd_ptrs.size());
    auto bn_ptr = std::move(bn_fwd_ptrs.at(config.index));

    result.invoker_factory = [args      = CKArgsBNormFwdTraining{bn_problem},
                              sh_bn_ptr = std::shared_ptr{std::move(bn_ptr)}](
                                 const std::vector<Kernel>& /*kernels*/) mutable {
        return [args = std::move(args), sh_bn_ptr = std::move(sh_bn_ptr)](
                   const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& params =
                primitive_parameters.CastTo<miopen::batchnorm::FwdTrainInvokeParams>();

            auto argument_ptr = args.MakeArgPtr(sh_bn_ptr, params);

            auto invoker_ptr            = sh_bn_ptr->MakeInvokerPointer();
            const auto enable_profiling = handle.IsProfilingEnabled();

            float elapsed_time =
                invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
            if(enable_profiling)
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed_time);
            }
        };
    };
    return result;
}
#endif

PerformanceConfigBnCKFwdTraining BnCKFwdTraining::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnCKFwdTraining pp;
    pp.HeuristicInit(problem_desc);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnCKFwdTraining::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnCKFwdTraining& config) const
{
    return config.IsValid(ctx, problem_desc);
}

PerformanceConfigBnCKFwdTraining
BnCKFwdTraining::Search(const ExecutionContext& ctx,
                        const miopen::batchnorm::ProblemDescription& problem_desc,
                        const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem_desc, invoke_ctx);
}

bool BnCKFwdTraining::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CK_BN_FWD_TRAINING))
        return false;
    if(!bn_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_supported_hardware(context.GetStream()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;

    switch(bn_problem.GetXDesc().GetType())
    {
    case miopenHalf: return (CheckCKApplicability<F16, F16, F32, F16, F16, F32>(bn_problem) != -1);
    case miopenBFloat16:
        return (CheckCKApplicability<BF16, BF16, F32, BF16, BF16, F32>(bn_problem) != -1);
    case miopenFloat: return (CheckCKApplicability<F32, F32, F32, F32, F32, F32>(bn_problem) != -1);
    case miopenDouble:
        return (CheckCKApplicability<F64, F64, F64, F64, F64, F64>(bn_problem) != -1);
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
#endif
    return false;
}

template <typename InvokerFactoryMakerNHWC>
ConvSolution MakeAnyInvokerFactory(const miopen::batchnorm::ProblemDescription& problem,
                                   InvokerFactoryMakerNHWC&& invoker_factory_maker_nhwc)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.IsLayoutNHWC())
    {
        switch(problem.GetXDesc().GetType())
        {
        case miopenFloat: return invoker_factory_maker_nhwc(F32{});
        case miopenDouble: return invoker_factory_maker_nhwc(F64{});
        case miopenHalf: return invoker_factory_maker_nhwc(F16{});
        case miopenBFloat16: return invoker_factory_maker_nhwc(BF16{});
        default:
            MIOPEN_THROW(miopenStatusInternalError,
                         "BnCKFwdInference operation does not support this data type");
        }
    }
    // Todo: problem.IsLayoutDefault()
    else
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "BnCKFwdInference operation does not support this data layout");
    }
#else
    return {};
#endif
}

ConvSolution BnCKFwdTraining::GetSolution(
    [[maybe_unused]] const ExecutionContext&,
    [[maybe_unused]] const miopen::batchnorm::ProblemDescription& bn_problem,
    [[maybe_unused]] const PerformanceConfigBnCKFwdTraining& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeAnyInvokerFactory(
        bn_problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);

            using AccTy = std::conditional_t<std::is_same_v<T, F64>,
                                             T,    // T==F64
                                             F32>; // T==F32
            return InvokerFactoryMakerNHWC<T, T, AccTy, T, T, AccTy>(bn_problem, config);
        }
        // Todo: InvokerFactoryMakerNCHW
    );
#else
    std::ignore = bn_problem;
    std::ignore = config;
    return {};
#endif
}

//=======================

} // namespace batchnorm
} // namespace solver
} // namespace miopen
