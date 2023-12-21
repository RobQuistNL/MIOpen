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

#include <miopen/reduce/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace reduce {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto xlength   = xDesc.GetLengths();
    auto ylength   = yDesc.GetLengths();
    auto dtype     = xDesc.GetType();
    auto sort_dims = dims;
    std::sort(sort_dims, sort_dims + (dims_size - 1));

    std::ostringstream ss;

    ss << "dtype" << dtype;
    for(int32_t i = 0; i < dims_size; i++)
    {
        int32_t dim       = sort_dims[i];
        auto reduce_size  = xlength[dim];
        auto output_numel = std::accumulate(
            ylength.begin(), ylength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

        ss << "dim" << dim;
        ss << "reduce_size" << reduce_size;
        ss << "output_numel" << output_numel;
    }
    return NetworkConfig{ss.str()};
}

} // namespace reduce

} // namespace miopen
