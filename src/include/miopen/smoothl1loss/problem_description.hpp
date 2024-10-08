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
#pragma once

#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace smoothl1loss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& oDesc_,
                              const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), reduction(reduction_)
    {
        IsSameType();
        IsSameLength();
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }
    miopenLossReductionMode_t GetReduction() const { return reduction; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Target tensor types do not match.");
        if(iDesc.GetType() != oDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Output tensor types do not match.");
        return true;
    }

    bool IsSameLength() const
    {
        if(iDesc.GetLengths() != tDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Target tensor dimension lengths do not match.");
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE && iDesc.GetLengths() != oDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Without reduction, Input and Output tensor dimension "
                         "lengths should be equal.");
        if(reduction != MIOPEN_LOSS_REDUCTION_NONE && oDesc.GetElementSize() != 1)
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SmoothL1Loss: When reduction, Output tensor dimension lengths must be (1).");
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
    miopenLossReductionMode_t reduction;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct BackwardProblemDescription : ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& iDesc_,
                               const TensorDescriptor& tDesc_,
                               const TensorDescriptor& dODesc_,
                               const TensorDescriptor& dIDesc_,
                               const TensorDescriptor& dTDesc_,
                               const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_),
          tDesc(tDesc_),
          dODesc(dODesc_),
          dIDesc(dIDesc_),
          dTDesc(dTDesc_),
          reduction(reduction_)
    {
        IsSameType();
        IsSameLength();
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetDODesc() const { return dODesc; }
    const TensorDescriptor& GetDIDesc() const { return dIDesc; }
    const TensorDescriptor& GetDTDesc() const { return dTDesc; }
    miopenLossReductionMode_t GetReduction() const { return reduction; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Target tensor types do not match.");
        if(iDesc.GetType() != dIDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and its Gradient tensor types do not match.");
        if(tDesc.GetType() != dTDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Target and its Gradient tensor types do not match.");
        if(iDesc.GetType() != dODesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Output Gradient tensor types do not match.");
        return true;
    }

    bool IsSameLength() const
    {
        if(iDesc.GetLengths() != tDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: Input and Target tensor dimension lengths do not match.");
        if(iDesc.GetLengths() != dIDesc.GetLengths())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SmoothL1Loss: Input and its Gradient tensor dimension lengths do not match.");
        if(tDesc.GetLengths() != dTDesc.GetLengths())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SmoothL1Loss: Target and its Gradient tensor dimension lengths do not match.");
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE && iDesc.GetLengths() != dODesc.GetLengths())
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SmoothL1Loss: Without reduction, Input and Output Gradient tensor dimension "
                "lengths should be equal.");
        if(reduction != MIOPEN_LOSS_REDUCTION_NONE && dODesc.GetElementSize() != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "SmoothL1Loss: When reduction, Output Gradient tensor dimension lengths "
                         "must be (1).");
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!iDesc.IsContiguous() || !tDesc.IsContiguous() || !dIDesc.IsContiguous() ||
           !dTDesc.IsContiguous() || !dODesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor dODesc;
    TensorDescriptor dIDesc;
    TensorDescriptor dTDesc;
    miopenLossReductionMode_t reduction;

    NetworkConfig MakeBackwardNetworkConfig() const;
};

} // namespace smoothl1loss

} // namespace miopen
