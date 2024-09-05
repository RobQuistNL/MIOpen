/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#define MIO_BN_TEST_EXPAVGFACTOR 0.99
#define MIO_BN_TEST_EPSILON 1e-5 // FLT_EPSILON
#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

#include "test.hpp"
#include "driver.hpp"
#include "fusionHost.hpp"
#include <miopen/stringutils.hpp>

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);

using ManagedFindOptions = std::unique_ptr<std::remove_pointer_t<miopenFindOptions_t>,
                                           miopenStatus_t (*)(miopenFindOptions_t)>;

ptr_FusionPlanDesc GetManagedFusionPlanDesc(miopenTensorDescriptor_t inputDesc)
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
    return ptr_FusionPlanDesc{fusePlanDesc};
}

ptr_FusionPlanArgs GetManageFusionPlanArgs()
{
    miopenOperatorArgs_t fusionArgs;
    miopenCreateOperatorArgs(&fusionArgs);
    return ptr_FusionPlanArgs{fusionArgs};
}

ptr_ActivationDesc GetManagedActivDesc()
{
    miopenActivationDescriptor_t activdesc;
    miopenCreateActivationDescriptor(&activdesc);
    return ptr_ActivationDesc{activdesc};
}

static inline ManagedFindOptions MakeFindOtions()
{
    miopenFindOptions_t find_options_raw;
    miopenCreateFindOptions(&find_options_raw);
    return {find_options_raw, &miopenDestroyFindOptions};
}

template <class T, class U>
struct verify_fwd_batchnorm_spatial_activ
{
    tensor<T> x;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenProblem_t problem;
    // miopenFusionPlanDescriptor_t fusionplan;
    // miopenFusionOpDescriptor_t bNormFwdOp;
    // miopenFusionOpDescriptor_t activFwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    double alpha;
    double beta;

    verify_fwd_batchnorm_spatial_activ( // miopenFusionPlanDescriptor_t pfwdfusionplan,
        miopenProblem_t problem_,
        const tensor<T>& pinput,
        miopenActivationDescriptor_t pactivDesc,
        const tensor<U>& pbnscale,
        const tensor<U>& pbnbias/*,
        miopenFusionOpDescriptor_t pbNormFwdOp,
        miopenFusionOpDescriptor_t pactivFwdOp*/)
    {
        problem   = problem_;
        x         = pinput;
        activDesc = pactivDesc;
        bnscale   = pbnscale;
        bnbias    = pbnbias;
        // fusionplan   = pfwdfusionplan;
        // bNormFwdOp   = pbNormFwdOp;
        // activFwdOp   = pactivFwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNSpatial);
        ssn = ssc = ssh = ssw        = 0;
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
        alpha                        = 1.;
        beta                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {

        auto bout = x;
        std::fill(bout.begin(), bout.end(), 0.);
        auto aout = x;
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        batchNormSpatialHostFwdTrain(x,
                                     bout,
                                     bnscale,
                                     bnbias,
                                     epsilon,
                                     expAvgFactor,
                                     savedMean,
                                     savedInvVar,
                                     runMean,
                                     runVar);
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        return std::make_tuple(aout, runMean, runVar, savedMean, savedInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu()
    {
        auto&& handle = get_handle();

        auto baout = x;
        std::fill(baout.begin(), baout.end(), 0.);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        auto in_dev              = handle.Write(x.data);
        auto out_dev             = handle.Write(baout.data);
        auto bnscale_dev         = handle.Write(bnscale.data);
        auto bnbias_dev          = handle.Write(bnbias.data);
        auto savedMean_dev       = handle.Write(savedMean.data);
        auto savedInvVar_dev     = handle.Write(savedInvVar.data);
        auto runningMean_dev     = handle.Write(runMean.data);
        auto runningVariance_dev = handle.Write(runVar.data);

        const auto find_options = MakeFindOtions();
        // clang-format off
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormX, in_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScale, bnscale_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedMean, savedMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedVariance, savedInvVar_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormRunningMean, runningMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormRunningVariance, runningVariance_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBias, bnbias_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationY, out_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenScalarBatchnormExpAvgFactor, &expAvgFactor);
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenScalarBatchnormEpsilon, &epsilon);
        // clang-format on

        const auto run_tensors = std::array{
            // clang-format off
            miopenTensorArgument_t{miopenTensorBatchnormX, nullptr, in_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScale, nullptr, bnscale_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedMean, nullptr, savedMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedVariance, nullptr, savedInvVar_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormRunningMean, nullptr, runningMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormRunningVariance, nullptr, runningVariance_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBias, nullptr, bnbias_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationY, nullptr, out_dev.get()},
            miopenTensorArgument_t{miopenScalarBatchnormExpAvgFactor, nullptr, &expAvgFactor},
            miopenTensorArgument_t{miopenScalarBatchnormEpsilon, nullptr, &epsilon},
            // clang-format on
        };

        /*double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormForward(ptr_fusionargs.get(),
                                        bNormFwdOp,
                                        &alpha,
                                        &beta,
                                        bnscale_dev.get(),
                                        bnbias_dev.get(),
                                        savedMean_dev.get(),
                                        savedInvVar_dev.get(),
                                        runningMean_dev.get(),
                                        runningVariance_dev.get(),
                                        expAvgFactor,
                                        epsilon);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activFwdOp, &alpha, &beta, activ_alpha, activ_beta,
        activ_gamma);

        auto lclxdesc = x.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lclxdesc,
                                in_dev.get(),
                                &lclxdesc,
                                out_dev.get(),
                                ptr_fusionargs.get());
                                */

        std::vector<miopenSolution_t> solutions;
        solutions.resize(1);

        {
            std::size_t solutions_found;
            const auto find_ret = miopenFindSolutions(&handle,
                                                      problem,
                                                      find_options.get(),
                                                      solutions.data(),
                                                      &solutions_found,
                                                      solutions.size());

            EXPECT_EQUAL(find_ret, miopenStatusSuccess);
            solutions.resize(solutions_found);
        }

        for(const auto solution : solutions)
        {
            const auto run_ret = miopenRunSolution(
                &handle, solution, run_tensors.size(), run_tensors.data(), nullptr, 0);
            EXPECT_EQUAL(run_ret, miopenStatusSuccess);
        }

        baout.data       = handle.Read<T>(out_dev, baout.data.size());
        runMean.data     = handle.Read<U>(runningMean_dev, runMean.data.size());
        runVar.data      = handle.Read<U>(runningVariance_dev, runVar.data.size());
        savedMean.data   = handle.Read<U>(savedMean_dev, savedMean.data.size());
        savedInvVar.data = handle.Read<U>(savedInvVar_dev, savedInvVar.data.size());

        return std::make_tuple(baout, runMean, runVar, savedMean, savedInvVar);
    }

    void fail(int badtensor) const
    {

        std::cout << "Forward Train Spatial Batch Normalization + Activation: " << std::endl;
        std::cout << "Input tensor: " << x.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4):
            std::cout << "Saved Inverse Variance tensor failed verification." << std::endl;
            break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_bwd_batchnorm_spatial_activ
{
    tensor<T> x;
    tensor<T> y;
    tensor<T> dy;
    tensor<U> savedMean;
    tensor<U> savedInvVar;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenProblem_t problem;
    // miopenFusionPlanDescriptor_t fusionplan;
    // miopenFusionOpDescriptor_t bNormBwdOp;
    // miopenFusionOpDescriptor_t activBwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    std::size_t input_n, input_c, input_h, input_w;
    double alpha;
    double beta;

    verify_bwd_batchnorm_spatial_activ(miopenProblem_t problem_,
                                       // miopenFusionPlanDescriptor_t pbwdfusionplan,
                                       const tensor<T>& pdyin,
                                       const tensor<T>& pxin,
                                       const tensor<T>& pyin,
                                       miopenActivationDescriptor_t pactivDesc,
                                       const tensor<U>& pbnscale,
                                       const tensor<U>& pbnbias,
                                       const tensor<U>& psavedMean,
                                       const tensor<U>& psavedInvVar/*,
                                       miopenFusionOpDescriptor_t pbNormBwdOp,
                                       miopenFusionOpDescriptor_t pactivBwdOp*/)
    {
        x           = pxin;
        y           = pyin;
        dy          = pdyin;
        savedMean   = psavedMean;
        savedInvVar = psavedInvVar;
        activDesc   = pactivDesc;
        bnscale     = pbnscale;
        bnbias      = pbnbias;
        problem     = problem_;
        // fusionplan   = pbwdfusionplan;
        // bNormBwdOp   = pbNormBwdOp;
        // activBwdOp   = pactivBwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNSpatial);
        ssn = ssc = ssh = ssw        = 0;
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
        input_n = input_c = input_h = input_w        = 0;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
        auto dx = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto bout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(bout.begin(), bout.end(), 0.);

        auto aout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(aout.begin(), aout.end(), 0.);

        auto find_options = [&]() {
            miopenFindOptions_t find_options_raw;
            miopenCreateFindOptions(&find_options_raw);
            return ManagedFindOptions{find_options_raw, &miopenDestroyFindOptions};
        }();

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        batchNormActivSpatialHostBwdTrain(activ_mode,
                                          activ_gamma,
                                          activ_beta,
                                          activ_alpha,
                                          x,
                                          dy,
                                          dx,
                                          bnscale,
                                          y,
                                          bnbias,
                                          dgamma,
                                          dbeta,
                                          savedMean,
                                          savedInvVar);

        return std::make_tuple(dx, dgamma, dbeta);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();
        auto dx       = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        auto xin_dev         = handle.Write(x.data);
        auto dxout_dev       = handle.Write(dx.data);
        auto yin_dev         = handle.Write(y.data);
        auto dyin_dev        = handle.Write(dy.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);
        auto dgamma_dev      = handle.Write(dgamma.data);
        auto dbeta_dev       = handle.Write(dbeta.data);

        const auto find_options = MakeFindOtions();
        // clang-format off
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormX, xin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormDY, dyin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScale, bnscale_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedMean, savedMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedVariance, savedInvVar_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScaleDiff, dgamma_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBiasDiff, dbeta_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBias, bnbias_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationX, xin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationY, yin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationDX, dxout_dev.get());
        // clang-format on

        const auto run_tensors = std::array{
            // clang-format off
            miopenTensorArgument_t{miopenTensorBatchnormX, nullptr, xin_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormDY, nullptr, dyin_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScale, nullptr, bnscale_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedMean, nullptr, savedMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedVariance, nullptr, savedInvVar_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScaleDiff, nullptr, dgamma_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBiasDiff, nullptr, dbeta_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBias, nullptr, bnbias_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationX, nullptr, yin_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationY, nullptr, yin_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationDX, nullptr, dxout_dev.get()},
            // clang-format on
        };

        std::vector<miopenSolution_t> solutions;
        solutions.resize(1);

        {
            std::size_t solutions_found;
            const auto find_ret = miopenFindSolutions(&handle,
                                                      problem,
                                                      find_options.get(),
                                                      solutions.data(),
                                                      &solutions_found,
                                                      solutions.size());
            EXPECT_EQUAL(find_ret, miopenStatusSuccess);
            solutions.resize(solutions_found);
        }

        for(const auto solution : solutions)
        {
            const auto run_ret = miopenRunSolution(
                &handle, solution, run_tensors.size(), run_tensors.data(), nullptr, 0);
            EXPECT_EQUAL(run_ret, miopenStatusSuccess);
        }

        /*
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormBackward(ptr_fusionargs.get(),
                                         bNormBwdOp,
                                         &alpha,
                                         &beta,
                                         xin_dev.get(),
                                         bnscale_dev.get(),
                                         bnbias_dev.get(),
                                         dgamma_dev.get(),
                                         dbeta_dev.get(),
                                         savedMean_dev.get(),
                                         savedInvVar_dev.get());
        miopenSetOpArgsActivBackward(ptr_fusionargs.get(),
                                     activBwdOp,
                                     &alpha,
                                     &beta,
                                     yin_dev.get(),
                                     nullptr,
                                     activ_alpha,
                                     activ_beta,
                                     activ_gamma);

        auto lcldydesc = dy.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lcldydesc,
                                dyin_dev.get(),
                                &lcldydesc,
                                dxout_dev.get(),
                                ptr_fusionargs.get());
                                */

        dx.data     = handle.Read<T>(dxout_dev, dx.data.size());
        dgamma.data = handle.Read<U>(dgamma_dev, dgamma.data.size());
        dbeta.data  = handle.Read<U>(dbeta_dev, dbeta.data.size());
        return std::make_tuple(dx, dgamma, dbeta);
    }

    void fail(int badtensor) const
    {

        std::cout << "Backward Train Spatial Batch Normalization + Activation: " << std::endl;
        std::cout << "Input x tensor: " << x.desc.ToString() << std::endl;
        std::cout << "Input y tensor: " << y.desc.ToString() << std::endl;
        std::cout << "Input dy tensor: " << dy.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "dx output tensor failed verification." << std::endl; break;
        case(1): std::cout << "dgamma output tensor failed verification." << std::endl; break;
        case(2): std::cout << "dbeta output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_fwd_batchnorm_peract_activ
{
    tensor<T> x;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenProblem_t problem;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    std::size_t input_n, input_c, input_h, input_w;
    double alpha;
    double beta;

    verify_fwd_batchnorm_peract_activ(miopenProblem_t problem_,
                                      const tensor<T>& pinput,
                                      miopenActivationDescriptor_t pactivDesc,
                                      const tensor<U>& pbnscale,
                                      const tensor<U>& pbnbias)
    {
        x            = pinput;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        problem      = problem_;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNPerActivation);
        ssn = ssc = ssh = ssw = 0;
        input_n = input_c = input_h = input_w        = 0;
        std::tie(ssn, ssc, ssh, ssw)                 = miopen::tien<4>(derivedBnDesc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {
        auto bout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(bout.begin(), bout.end(), 0.);

        auto aout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        batchNormPerActHostFwdTrain(x,
                                    bout,
                                    bnscale,
                                    bnbias,
                                    epsilon,
                                    expAvgFactor,
                                    savedMean,
                                    savedInvVar,
                                    runMean,
                                    runVar);
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        return std::make_tuple(aout, runMean, runVar, savedMean, savedInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu()
    {
        auto&& handle = get_handle();
        auto baout    = x;
        std::fill(baout.begin(), baout.end(), 0.);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        auto in_dev              = handle.Write(x.data);
        auto out_dev             = handle.Write(baout.data);
        auto bnscale_dev         = handle.Write(bnscale.data);
        auto bnbias_dev          = handle.Write(bnbias.data);
        auto savedMean_dev       = handle.Write(savedMean.data);
        auto savedInvVar_dev     = handle.Write(savedInvVar.data);
        auto runningMean_dev     = handle.Write(runMean.data);
        auto runningVariance_dev = handle.Write(runVar.data);
        // double activ_alpha, activ_beta, activ_gamma;

        const auto find_options = MakeFindOtions();
        // clang-format off
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormX, in_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScale, bnscale_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedMean, savedMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedVariance, savedInvVar_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormRunningMean, runningMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormRunningVariance, runningVariance_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBias, bnbias_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationY, out_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenScalarBatchnormExpAvgFactor, &expAvgFactor);
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenScalarBatchnormEpsilon, &epsilon);
        // clang-format on

        const auto run_tensors = std::array{
            // clang-format off
            miopenTensorArgument_t{miopenTensorBatchnormX, nullptr, in_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScale, nullptr, bnscale_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedMean, nullptr, savedMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedVariance, nullptr, savedInvVar_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormRunningMean, nullptr, runningMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormRunningVariance, nullptr, runningVariance_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBias, nullptr, bnbias_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationY, nullptr, out_dev.get()},
            miopenTensorArgument_t{miopenScalarBatchnormExpAvgFactor, nullptr, &expAvgFactor},
            miopenTensorArgument_t{miopenScalarBatchnormEpsilon, nullptr, &epsilon},
            // clang-format on
        };

        std::vector<miopenSolution_t> solutions;
        solutions.resize(1);

        {
            std::size_t solutions_found;
            const auto find_ret = miopenFindSolutions(&handle,
                                                      problem,
                                                      find_options.get(),
                                                      solutions.data(),
                                                      &solutions_found,
                                                      solutions.size());

            EXPECT_EQUAL(find_ret, miopenStatusSuccess);
            solutions.resize(solutions_found);
        }

        for(const auto solution : solutions)
        {
            const auto run_ret = miopenRunSolution(
                &handle, solution, run_tensors.size(), run_tensors.data(), nullptr, 0);
            EXPECT_EQUAL(run_ret, miopenStatusSuccess);
        }

        // miopenActivationMode_t activ_mode;
        // miopenGetActivationDescriptor(
        //     activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        // auto ptr_fusionargs = GetManageFusionPlanArgs();
        // miopenSetOpArgsBatchNormForward(ptr_fusionargs.get(),
        //                                 bNormFwdOp,
        //                                 &alpha,
        //                                 &beta,
        //                                 bnscale_dev.get(),
        //                                 bnbias_dev.get(),
        //                                 savedMean_dev.get(),
        //                                 savedInvVar_dev.get(),
        //                                 runningMean_dev.get(),
        //                                 runningVariance_dev.get(),
        //                                 expAvgFactor,
        //                                 epsilon);
        // miopenSetOpArgsActivForward(
        //     ptr_fusionargs.get(), activFwdOp, &alpha, &beta, activ_alpha, activ_beta,
        //     activ_gamma);

        // auto lclxdesc = x.desc;
        // miopenExecuteFusionPlan(&handle,
        //                         fusionplan,
        //                         &lclxdesc,
        //                         in_dev.get(),
        //                         &lclxdesc,
        //                         out_dev.get(),
        //                         ptr_fusionargs.get());

        baout.data       = handle.Read<T>(out_dev, baout.data.size());
        runMean.data     = handle.Read<U>(runningMean_dev, runMean.data.size());
        runVar.data      = handle.Read<U>(runningVariance_dev, runVar.data.size());
        savedMean.data   = handle.Read<U>(savedMean_dev, savedMean.data.size());
        savedInvVar.data = handle.Read<U>(savedInvVar_dev, savedInvVar.data.size());

        return std::make_tuple(baout, runMean, runVar, savedMean, savedInvVar);
    }

    void fail(int badtensor) const
    {

        std::cout << "Forward Train Per Activation Batch Normalization + Activation: " << std::endl;
        std::cout << "Input tensor: " << x.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4):
            std::cout << "Saved Inverse Variance tensor failed verification." << std::endl;
            break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_bwd_batchnorm_peract_activ
{
    tensor<T> x;
    tensor<T> y;
    tensor<T> dy;
    tensor<U> savedMean;
    tensor<U> savedInvVar;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenProblem_t problem;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    double alpha;
    double beta;
    std::size_t input_n, input_c, input_h, input_w;

    verify_bwd_batchnorm_peract_activ(miopenProblem_t problem_,
                                      const tensor<T>& pdyin,
                                      const tensor<T>& pxin,
                                      const tensor<T>& pyin,
                                      miopenActivationDescriptor_t pactivDesc,
                                      const tensor<U>& pbnscale,
                                      const tensor<U>& pbnbias,
                                      const tensor<U>& psavedMean,
                                      const tensor<U>& psavedInvVar)
    {
        x            = pxin;
        y            = pyin;
        dy           = pdyin;
        savedMean    = psavedMean;
        savedInvVar  = psavedInvVar;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        problem      = problem_;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNPerActivation);
        ssn = ssc = ssh = ssw = 0;
        input_n = input_c = input_h = input_w        = 0;
        std::tie(ssn, ssc, ssh, ssw)                 = miopen::tien<4>(derivedBnDesc.GetLengths());
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

        auto dx = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        batchNormActivPerActHostBwdTrain(activ_mode,
                                         activ_gamma,
                                         activ_beta,
                                         activ_alpha,
                                         x,
                                         dy,
                                         y,
                                         dx,
                                         bnscale,
                                         bnbias,
                                         dgamma,
                                         dbeta,
                                         savedMean,
                                         savedInvVar);

        return std::make_tuple(dx, dgamma, dbeta);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();
        auto dx       = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        auto xin_dev         = handle.Write(x.data);
        auto dxout_dev       = handle.Write(dx.data);
        auto yin_dev         = handle.Write(y.data);
        auto dyin_dev        = handle.Write(dy.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);
        auto dgamma_dev      = handle.Write(dgamma.data);
        auto dbeta_dev       = handle.Write(dbeta.data);

        const auto find_options = MakeFindOtions();
        // clang-format off
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormX, xin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormDY, dyin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScale, bnscale_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedMean, savedMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormSavedVariance, savedInvVar_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScaleDiff, dgamma_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBiasDiff, dbeta_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBias, bnbias_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationX, xin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationY, yin_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationDX, dxout_dev.get());
        // clang-format on

        const auto run_tensors = std::array{
            // clang-format off
            miopenTensorArgument_t{miopenTensorBatchnormX, nullptr, xin_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormDY, nullptr, dyin_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScale, nullptr, bnscale_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedMean, nullptr, savedMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormSavedVariance, nullptr, savedInvVar_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScaleDiff, nullptr, dgamma_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBiasDiff, nullptr, dbeta_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBias, nullptr, bnbias_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationX, nullptr, yin_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationY, nullptr, yin_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationDX, nullptr, dxout_dev.get()},
            // clang-format on
        };

        std::vector<miopenSolution_t> solutions;
        solutions.resize(1);

        {
            std::size_t solutions_found;
            const auto find_ret = miopenFindSolutions(&handle,
                                                      problem,
                                                      find_options.get(),
                                                      solutions.data(),
                                                      &solutions_found,
                                                      solutions.size());
            EXPECT_EQUAL(find_ret, miopenStatusSuccess);
            solutions.resize(solutions_found);
        }

        for(const auto solution : solutions)
        {
            const auto run_ret = miopenRunSolution(
                &handle, solution, run_tensors.size(), run_tensors.data(), nullptr, 0);
            EXPECT_EQUAL(run_ret, miopenStatusSuccess);
        }

        // double activ_alpha, activ_beta, activ_gamma;
        // miopenActivationMode_t activ_mode;
        // miopenGetActivationDescriptor(
        //    activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        // auto ptr_fusionargs = GetManageFusionPlanArgs();
        // miopenSetOpArgsBatchNormBackward(ptr_fusionargs.get(),
        //                                 bNormBwdOp,
        //                                 &alpha,
        //                                 &beta,
        //                                 xin_dev.get(),
        //                                 bnscale_dev.get(),
        //                                 bnbias_dev.get(),
        //                                 dgamma_dev.get(),
        //                                 dbeta_dev.get(),
        //                                 savedMean_dev.get(),
        //                                 savedInvVar_dev.get());
        // miopenSetOpArgsActivBackward(ptr_fusionargs.get(),
        //                             activBwdOp,
        //                             &alpha,
        //                             &beta,
        //                             yin_dev.get(),
        //                             nullptr,
        //                             activ_alpha,
        //                             activ_beta,
        //                             activ_gamma);

        // auto lcldydesc = dy.desc;
        // miopenExecuteFusionPlan(&handle,
        //                        fusionplan,
        //                        &lcldydesc,
        //                        dyin_dev.get(),
        //                        &lcldydesc,
        //                        dxout_dev.get(),
        //                        ptr_fusionargs.get());

        dx.data     = handle.Read<T>(dxout_dev, dx.data.size());
        dgamma.data = handle.Read<U>(dgamma_dev, dgamma.data.size());
        dbeta.data  = handle.Read<U>(dbeta_dev, dbeta.data.size());
        return std::make_tuple(dx, dgamma, dbeta);
    }

    void fail(int badtensor) const
    {

        std::cout << "Backward Train Per Activation Batch Normalization + Activation: "
                  << std::endl;
        std::cout << "Input x tensor: " << x.desc.ToString() << std::endl;
        std::cout << "Input y tensor: " << y.desc.ToString() << std::endl;
        std::cout << "Input dy tensor: " << dy.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "dx output tensor failed verification." << std::endl; break;
        case(1): std::cout << "dgamma output tensor failed verification." << std::endl; break;
        case(2): std::cout << "dbeta output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

static std::string transform_mode(std::string s)
{
    return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
}

static inline void AddAndFuse(miopenProblem_t left,
                              std::function<void(miopenProblem_t*)>&& make_right_problem)
{
    miopenProblem_t right;
    make_right_problem(&right);
    EXPECT_EQUAL(miopenStatusSuccess, miopenFuseProblems(left, right));
    EXPECT_EQUAL(miopenStatusSuccess, miopenDestroyProblem(right));
};

template <class T>
struct na_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    ptr_ActivationDesc ptr_activdesc = nullptr;

    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::string amode;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 1;

    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;
    double alpha = 0., beta = 0., gamma = 0.;

    na_fusion_driver()
    {
        this->batch_factor = 4;

        add(batchnormMode, "batch-norm-mode", generate_data({/*0, */ 1}));
        add(input,
            "input",
            (batchnormMode == 1) ? get_bn_spatial_input_tensor(tensor_elem_gen_integer{max_value})
                                 : get_bn_peract_input_tensor(tensor_elem_gen_integer{max_value}));
        add(alpha, "alpha", generate_data({/*1.,*/ 0.5}));
        add(beta, "beta", generate_data({/*0.,*/ 0.5}));
        add(gamma, "gamma", generate_data({/*1.,*/ 0.5}));
        add(amode,
            "amode",
            generate_data(
                {"MIOPENACTIVATIONRELU" /*, "MIOPENACTIVATIONLOGISTIC", "MIOPENACTIVATIONABS"*/}));
    }

    void run()
    {
        amode = transform_mode(amode);

        // NOLINTBEGIN(*-braces-around-statements)
        if(amode == "PASSTHRU")
            activ_mode = miopenActivationPASTHRU;
        else if(amode == "LOGISTIC")
            activ_mode = miopenActivationLOGISTIC;
        else if(amode == "TANH")
            activ_mode = miopenActivationTANH;
        else if(amode == "RELU")
            activ_mode = miopenActivationRELU;
        else if(amode == "SOFTRELU")
            activ_mode = miopenActivationSOFTRELU;
        else if(amode == "ABS")
            activ_mode = miopenActivationABS;
        else if(amode == "POWER")
            activ_mode = miopenActivationPOWER;
        else if(amode == "CLIPPEDRELU")
            activ_mode = miopenActivationCLIPPEDRELU;
        else if(amode == "LEAKYRELU")
            activ_mode = miopenActivationLEAKYRELU;
        else if(amode == "ELU")
            activ_mode = miopenActivationELU;
        // NOLINTEND(*-braces-around-statements)

        std::size_t input_n, input_c, input_h, input_w;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());
        this->tolerance                              = 80 * double(input.desc.GetElementSize());
        ptr_activdesc                                = GetManagedActivDesc();
        miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);
        auto&& handle = get_handle();

        using ManagedProblem = std::unique_ptr<std::remove_pointer_t<miopenProblem_t>,
                                               miopenStatus_t (*)(miopenProblem_t)>;

        std::size_t ssn, ssc, ssh, ssw;
        if(batchnormMode == 1)
        {

            bnmode = miopenBNSpatial;

            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
            std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
            scale =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            shift =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            /*miopenCreateOpBatchNormForward(ptr_fwdfusionplan.get(), &bNormFwdOp, bnmode, true);
            miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
            miopenStatus_t miopenFwdError =
                miopenCompileFusionPlan(&handle, ptr_fwdfusionplan.get());
            if(miopenFwdError != miopenStatusSuccess)
            {
                std::cerr << "BatchNorm+Activation Spatial Forward Training plan not supported."
                          << std::endl;
                return;
            }*/

            const auto fwd_problem = [&]() {
                miopenProblem_t problem;
                miopenCreateBatchnormProblem(&problem, bnmode, true, miopenProblemDirectionForward);

                // clang-format off
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormX, &input.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScale, &scale.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedMean, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedVariance, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormRunningMean, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormRunningVariance, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBias, &shift.desc);
                // clang-format on

                AddAndFuse(problem, [&](auto activation) {
                    miopenCreateActivationProblem(
                        activation, ptr_activdesc.get(), miopenProblemDirectionForward);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationX, &input.desc);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationY, &input.desc);
                });

                return ManagedProblem{problem, &miopenDestroyProblem};
            }();

            const auto bwd_problem = [&]() {
                miopenProblem_t problem;
                miopenCreateBatchnormProblem(
                    &problem, bnmode, true, miopenProblemDirectionBackward);

                // clang-format off
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormDY, &input.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormX, &input.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScale, &scale.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedMean, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedVariance, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScaleDiff, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBiasDiff, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBias, &shift.desc);
                // clang-format on

                AddAndFuse(problem, [&](auto activation) {
                    miopenCreateActivationProblem(
                        activation, ptr_activdesc.get(), miopenProblemDirectionBackward);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationX, &input.desc);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationY, &input.desc);
                });

                return ManagedProblem{problem, &miopenDestroyProblem};
            }();

            /*
            auto fwdTrain =
                verify(verify_fwd_batchnorm_spatial_activ<T, PREC_TYPE>{ptr_fwdfusionplan.get(),
                                                                        input,
                                                                        ptr_activdesc.get(),
                                                                        scale,
                                                                        shift,
                                                                        bNormFwdOp,
                                                                        activFwdOp});
                                                                        */

            auto fwdTrain = verify(verify_fwd_batchnorm_spatial_activ<T, PREC_TYPE>{
                fwd_problem.get(), input, ptr_activdesc.get(), scale, shift});

            //        Tuple returns: (aout, runMean, runVar, savedMean, savedInvVar);
            auto y_in        = std::get<0>(fwdTrain.second);
            auto savedMean   = std::get<3>(fwdTrain.second);
            auto savedInvVar = std::get<4>(fwdTrain.second);
            auto dyin        = tensor<T>{input_n, input_c, input_h, input_w}.generate(
                tensor_elem_gen_integer{max_value});

            /*
            miopenCreateOpBatchNormBackward(ptr_bwdfusionplan.get(), &bNormBwdOp, bnmode);
            miopenCreateOpActivationBackward(ptr_bwdfusionplan.get(), &activBwdOp, activ_mode);
            miopenStatus_t miopenBwdError =
                miopenCompileFusionPlan(&handle, ptr_bwdfusionplan.get());
            if(miopenBwdError != miopenStatusSuccess)
            {
                std::cerr << "BatchNorm+Activation Spatial Backward Training plan not supported."
                          << std::endl;
                return;
            }
            */

            verify(verify_bwd_batchnorm_spatial_activ<T, PREC_TYPE>{
                bwd_problem.get(),
                // ptr_bwdfusionplan.get(),
                dyin,
                input,
                y_in,
                ptr_activdesc.get(),
                scale,
                shift,
                savedMean,
                savedInvVar,
                // bNormBwdOp,
                // activBwdOp,
            });
        }
        else if(batchnormMode == 0)
        {
            // miopenFusionOpDescriptor_t bNormFwdOp = nullptr;
            // miopenFusionOpDescriptor_t activFwdOp = nullptr;
            // auto ptr_fwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);

            miopenFusionOpDescriptor_t bNormBwdOp = nullptr;
            miopenFusionOpDescriptor_t activBwdOp = nullptr;
            auto ptr_bwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);

            bnmode = miopenBNPerActivation;
            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
            std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
            scale =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            shift =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            // miopenCreateOpBatchNormForward(ptr_fwdfusionplan.get(), &bNormFwdOp, bnmode, true);
            // miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
            // miopenStatus_t miopenFwdError =
            // miopenCompileFusionPlan(&handle, ptr_fwdfusionplan.get());
            // if(miopenFwdError != miopenStatusSuccess)
            // {
            //    std::cerr
            //        << "BatchNorm+Activation Per Activation Forward Training plan not supported."
            //        << std::endl;
            //    return;
            //}

            const auto fwd_problem = [&]() {
                miopenProblem_t problem;
                EXPECT_EQUAL(miopenStatusSuccess,
                             miopenCreateBatchnormProblem(
                                 &problem, bnmode, true, miopenProblemDirectionForward));

                // clang-format off
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormX, &input.desc));
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScale, &scale.desc));
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedMean, &derivedBnDesc));
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedVariance, &derivedBnDesc));
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormRunningMean, &derivedBnDesc));
                EXPECT_EQUAL(miopenStatusSuccess, miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormRunningVariance, &derivedBnDesc));
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBias, &shift.desc);
                // clang-format on

                AddAndFuse(problem, [&](auto activation) {
                    EXPECT_EQUAL(miopenStatusSuccess,
                                 miopenCreateActivationProblem(activation,
                                                               ptr_activdesc.get(),
                                                               miopenProblemDirectionForward));
                    EXPECT_EQUAL(miopenStatusSuccess,
                                 miopenSetProblemTensorDescriptor(
                                     *activation, miopenTensorActivationX, &input.desc));
                    EXPECT_EQUAL(miopenStatusSuccess,
                                 miopenSetProblemTensorDescriptor(
                                     *activation, miopenTensorActivationY, &input.desc));
                });

                return ManagedProblem{problem, &miopenDestroyProblem};
            }();

            const auto bwd_problem = [&]() {
                miopenProblem_t problem;
                EXPECT_EQUAL(miopenStatusSuccess,
                             miopenCreateBatchnormProblem(
                                 &problem, bnmode, true, miopenProblemDirectionBackward));

                // clang-format off
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormDY, &input.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormX, &input.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScale, &scale.desc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedMean, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormSavedVariance, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScaleDiff, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBiasDiff, &derivedBnDesc);
                miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBias, &shift.desc);
                // clang-format on

                AddAndFuse(problem, [&](auto activation) {
                    miopenCreateActivationProblem(
                        activation, ptr_activdesc.get(), miopenProblemDirectionBackward);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationX, &input.desc);
                    miopenSetProblemTensorDescriptor(
                        *activation, miopenTensorActivationY, &input.desc);
                });

                return ManagedProblem{problem, &miopenDestroyProblem};
            }();

            auto fwdTrain    = verify(verify_fwd_batchnorm_peract_activ<T, PREC_TYPE>{
                fwd_problem.get(), input, ptr_activdesc.get(), scale, shift});
            auto y_in        = std::get<0>(fwdTrain.second);
            auto savedMean   = std::get<3>(fwdTrain.second);
            auto savedInvVar = std::get<4>(fwdTrain.second);
            auto dyin        = tensor<T>{input_n, input_c, input_h, input_w}.generate(
                tensor_elem_gen_integer{max_value});

            miopenCreateOpBatchNormBackward(ptr_bwdfusionplan.get(), &bNormBwdOp, bnmode);
            miopenCreateOpActivationBackward(ptr_bwdfusionplan.get(), &activBwdOp, activ_mode);
            miopenStatus_t miopenBwdError =
                miopenCompileFusionPlan(&handle, ptr_bwdfusionplan.get());
            if(miopenBwdError != miopenStatusSuccess)
            {
                std::cerr
                    << "BatchNorm+Activation Per Activation Backward Training plan not supported."
                    << std::endl;
                return;
            }
            verify(verify_bwd_batchnorm_peract_activ<T, PREC_TYPE>{bwd_problem.get(),
                                                                   dyin,
                                                                   input,
                                                                   y_in,
                                                                   ptr_activdesc.get(),
                                                                   scale,
                                                                   shift,
                                                                   savedMean,
                                                                   savedInvVar});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<na_fusion_driver>(argc, argv); }
