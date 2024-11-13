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
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <chrono>
#include "../../driver/random.hpp"

#include <gtest/gtest.h>

namespace {
[[gnu::noinline]] size_t old_GetElementSpace(const std::vector<size_t>& lens,
                                             const std::vector<size_t>& strides,
                                             size_t vector_length)
{
    std::vector<size_t> maxIndices(lens.size());
    std::transform(lens.begin(),
                   lens.end(),
                   std::vector<size_t>(lens.size(), 1).begin(),
                   maxIndices.begin(),
                   std::minus<size_t>());
    return std::inner_product(maxIndices.begin(), maxIndices.end(), strides.begin(), size_t{0}) +
           vector_length;
}

[[gnu::noinline]] size_t new_GetElementSpace(const std::vector<size_t>& lens,
                                             const std::vector<size_t>& strides,
                                             size_t vector_length)
{
    return std::inner_product(lens.begin(),
                              lens.end(),
                              strides.begin(),
                              vector_length,
                              std::plus<size_t>(),
                              [](size_t len, size_t stride) { return (len - 1) * stride; });
}
} // namespace

TEST(CPU_GetElementSpace_NONE, CompareGetElementSpaceFwd)
{
    size_t new_time = 0;
    size_t old_time = 0;

    static constexpr size_t cnt = 1024 * 1024 * 64;

    for(int i = 0; i < cnt; ++i)
    {
        size_t dims = prng::gen_0_to_B<size_t>(4) + 1; // 1d-5d
        std::vector<size_t> lens(dims);
        std::vector<size_t> strides(dims);
        size_t vector_length = prng::gen_0_to_B<size_t>(7) + 1; // 1-8
        for(size_t d = 0; d < dims; ++d)
        {
            lens[d]    = prng::gen_0_to_B<size_t>(200) + vector_length;
            strides[d] = prng::gen_0_to_B<size_t>(200) + vector_length;
        }

        auto t1         = std::chrono::high_resolution_clock::now();
        size_t old_func = old_GetElementSpace(lens, strides, vector_length);
        auto t2         = std::chrono::high_resolution_clock::now();
        size_t new_func = new_GetElementSpace(lens, strides, vector_length);
        auto t3         = std::chrono::high_resolution_clock::now();
        ASSERT_EQ(new_func, old_func);
        old_time += std::chrono::high_resolution_clock::duration(t2 - t1).count();
        new_time += std::chrono::high_resolution_clock::duration(t3 - t2).count();
    }
    std::cout << "Number of tests: " << cnt
              << "\nNew function average time (ns): " << new_time / double(cnt)
              << "\nOld function average time (ns): " << old_time / double(cnt)
              << "\nGain (times): " << old_time / double(new_time) << std::endl;
};

TEST(CPU_GetElementSpace_NONE, CompareGetElementSpaceBwd)
{
    size_t new_time = 0;
    size_t old_time = 0;

    static constexpr size_t cnt = 1024 * 1024 * 64;

    for(int i = 0; i < cnt; ++i)
    {
        const size_t dims = prng::gen_0_to_B<size_t>(4) + 1; // 1d-5d
        std::vector<size_t> lens(dims);
        std::vector<size_t> strides(dims);
        size_t vector_length = prng::gen_0_to_B<size_t>(7) + 1; // 1-8
        for(size_t d = 0; d < dims; ++d)
        {
            lens[d]    = prng::gen_0_to_B<size_t>(200) + vector_length;
            strides[d] = prng::gen_0_to_B<size_t>(200) + vector_length;
        }

        auto t1         = std::chrono::high_resolution_clock::now();
        size_t new_func = new_GetElementSpace(lens, strides, vector_length);
        auto t2         = std::chrono::high_resolution_clock::now();
        size_t old_func = old_GetElementSpace(lens, strides, vector_length);
        auto t3         = std::chrono::high_resolution_clock::now();
        ASSERT_EQ(new_func, old_func);
        new_time += std::chrono::high_resolution_clock::duration(t2 - t1).count();
        old_time += std::chrono::high_resolution_clock::duration(t3 - t2).count();
    }
    std::cout << "Number of tests: " << cnt
              << "\nNew function average time (ns): " << new_time / double(cnt)
              << "\nOld function average time (ns): " << old_time / double(cnt)
              << "\nGain (times): " << old_time / double(new_time) << std::endl;
};
