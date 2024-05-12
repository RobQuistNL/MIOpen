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

#include "../driver/tensor_driver.hpp"
#include "cpu_smooth_l1loss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/smooth_l1loss.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct SmoothL1LossTestCase
{
    std::vector<size_t> lengths;
    float beta;
    float divisor;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const SmoothL1LossTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " Beta:" << tc.beta << " Divisor:" << tc.divisor
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<SmoothL1LossTestCase> SmoothL1LossTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs;
    tcs.push_back({{1, 2, 3, 4}, 1, 1, false});
    tcs.push_back({{1, 1, 1, 257}, 1, 1, false});
    tcs.push_back({{2, 10, 128, 128}, 1, 1, false});
    tcs.push_back({{5, 13, 17, 11}, 1, 1, false});
    tcs.push_back({{256, 4, 8723}, 1, 1, false});
    tcs.push_back({{1, 1, 1}, 1, 1, true});
    tcs.push_back({{34, 4, 5}, 1, 1, true});
    tcs.push_back({{4, 7, 5}, 1, 1, true});
    tcs.push_back({{15, 4, 5}, 1, 1, true});
    return tcs;
}

inline std::vector<size_t> GetStrides(std::vector<size_t> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<size_t> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T = float>
struct SmoothL1LossTestForward : public ::testing::TestWithParam<SmoothL1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        smooth_l1loss_config = GetParam();
        auto gen_value1      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 1); };
        auto gen_value2      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 2); };

        beta            = smooth_l1loss_config.beta;
        divisor         = smooth_l1loss_config.divisor;
        auto lengths    = smooth_l1loss_config.lengths;
        auto contiguous = smooth_l1loss_config.contiguous;

        auto in_strides = GetStrides(lengths, true);
        input           = tensor<T>{lengths, in_strides}.generate(gen_value1);

        auto tar_strides = GetStrides(lengths, contiguous);
        target           = tensor<T>{lengths, tar_strides}.generate(gen_value2);

        auto out_lengths = std::isnan(divisor) ? lengths : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_lengths, true);

        output = tensor<T>{out_lengths, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_lengths, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = std::isnan(divisor) ? 0
                                             : miopen::GetSmoothL1LossReducedForwardWorkspaceSize(
                                                   handle, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), 0.0f);

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), 0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(!std::isnan(divisor))
        {
            cpu_smooth_l1loss_reduced_forward<T>(
                input, target, ref_output, ref_workspace, beta, divisor);
            status         = miopen::SmoothL1LossReducedForward(handle,
                                                        workspace_dev.get(),
                                                        ws_sizeInBytes,
                                                        input.desc,
                                                        input_dev.get(),
                                                        target.desc,
                                                        target_dev.get(),
                                                        output.desc,
                                                        output_dev.get(),
                                                        beta,
                                                        divisor);
            workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error_w = miopen::rms_range(ref_workspace, workspace);

        EXPECT_TRUE(miopen::range_distance(ref_workspace) == miopen::range_distance(workspace));
        EXPECT_TRUE(error_w < tolerance);

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < tolerance)
            << "Error output beyond tolerance Error: " << error << ",  Tolerance: " << tolerance;
    }
    SmoothL1LossTestCase smooth_l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> workspace;

    tensor<T> ref_workspace;
    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    float beta;
    float divisor;
};

template <typename T = float>
struct SmoothL1LossTestBackward : public ::testing::TestWithParam<SmoothL1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        smooth_l1loss_config = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        beta            = smooth_l1loss_config.beta;
        divisor         = smooth_l1loss_config.divisor;
        auto lengths    = smooth_l1loss_config.lengths;
        auto contiguous = smooth_l1loss_config.contiguous;

        if(contiguous)
            GTEST_SKIP();

        auto in_strides = GetStrides(lengths, true);
        input           = tensor<T>{lengths, in_strides}.generate(gen_value1);

        auto tar_strides = GetStrides(lengths, contiguous);
        target           = tensor<T>{lengths, tar_strides}.generate(gen_value2);

        auto out_lengths = std::isnan(divisor) ? lengths : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_lengths, true);

        dO = tensor<T>{out_lengths, out_strides};
        std::fill(dO.begin(), dO.end(), 0.5);

        dI = tensor<T>{lengths, in_strides};
        std::fill(dI.begin(), dI.end(), std::numeric_limits<T>::quiet_NaN());
        dT = tensor<T>{lengths, tar_strides};
        std::fill(dT.begin(), dT.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dI = tensor<T>{lengths, in_strides};
        std::fill(ref_dI.begin(), ref_dI.end(), std::numeric_limits<T>::quiet_NaN());
        ref_dT = tensor<T>{lengths, tar_strides};
        std::fill(ref_dT.begin(), ref_dT.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        dO_dev     = handle.Write(dO.data);
        dI_dev     = handle.Write(dI.data);
        dT_dev     = handle.Write(dT.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(!std::isnan(divisor))
        {
            cpu_smooth_l1loss_reduced_backward<T>(input, target, dO, ref_dI, ref_dT, beta, divisor);
            status = miopen::SmoothL1LossReducedBackward(handle,
                                                         input.desc,
                                                         input_dev.get(),
                                                         target.desc,
                                                         target_dev.get(),
                                                         dO.desc,
                                                         dO_dev.get(),
                                                         dI.desc,
                                                         dI_dev.get(),
                                                         dT.desc,
                                                         dT_dev.get(),
                                                         beta,
                                                         divisor);
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        dI.data = handle.Read<T>(dI_dev, dI.data.size());
        dT.data = handle.Read<T>(dT_dev, dT.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error_dI = miopen::rms_range(ref_dI, dI);
        auto error_dT = miopen::rms_range(ref_dT, dT);

        EXPECT_TRUE(miopen::range_distance(ref_dI) == miopen::range_distance(dI));
        EXPECT_TRUE(miopen::range_distance(ref_dT) == miopen::range_distance(dT));
        EXPECT_TRUE(error_dI < tolerance && error_dT < tolerance)
            << "Error output beyond tolerance Error: {" << error_dI << "," << error_dT
            << "},  Tolerance: " << tolerance;
    }
    SmoothL1LossTestCase smooth_l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> dO;
    tensor<T> dI;
    tensor<T> dT;

    tensor<T> ref_dI;
    tensor<T> ref_dT;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dI_dev;
    miopen::Allocator::ManageDataPtr dT_dev;

    float beta;
    float divisor;
};
