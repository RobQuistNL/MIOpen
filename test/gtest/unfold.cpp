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

#include "unfold.hpp"
#include "miopen/bfloat16.hpp"
#include "tensor_holder.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace unfold {

std::string GetFloatArg()
{
    const auto& tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct UnfoldForwardTestFloat32 : UnfoldFwdTest<float>
{
};

struct UnfoldForwardTestFloat16 : UnfoldFwdTest<half>
{
};

struct UnfoldForwardTestBFloat16 : UnfoldFwdTest<bfloat16>
{
};

struct UnfoldBackwardTestFloat32 : UnfoldBwdTest<float>
{
};

struct UnfoldBackwardTestFloat16 : UnfoldBwdTest<half>
{
};

struct UnfoldBackwardTestBFloat16 : UnfoldBwdTest<bfloat16>
{
};
}; // namespace unfold
using namespace unfold;
TEST_P(UnfoldForwardTestFloat32, UnfoldForwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldForwardTestSet,
                         UnfoldForwardTestFloat32,
                         testing::ValuesIn(UnfoldTestConfigs()));

TEST_P(UnfoldForwardTestFloat16, UnfoldForwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldForwardTestSet,
                         UnfoldForwardTestFloat16,
                         testing::ValuesIn(UnfoldTestConfigs()));

TEST_P(UnfoldForwardTestBFloat16, UnfoldForwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldForwardTestSet,
                         UnfoldForwardTestBFloat16,
                         testing::ValuesIn(UnfoldTestConfigs()));

TEST_P(UnfoldBackwardTestFloat32, UnfoldBackwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldBackwardTestSet,
                         UnfoldBackwardTestFloat32,
                         testing::ValuesIn(UnfoldTestConfigs()));

TEST_P(UnfoldBackwardTestFloat16, UnfoldBackwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldBackwardTestSet,
                         UnfoldBackwardTestFloat16,
                         testing::ValuesIn(UnfoldTestConfigs()));

TEST_P(UnfoldBackwardTestBFloat16, UnfoldBackwardTest)
{
    if(!MIOPEN_TEST_ALL ||
       (env::enabled(MIOPEN_TEST_ALL) && env::value(MIOPEN_TEST_FLOAT_ARG) == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(UnfoldBackwardTestSet,
                         UnfoldBackwardTestBFloat16,
                         testing::ValuesIn(UnfoldTestConfigs()));
