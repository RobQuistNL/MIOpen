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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename T1, typename T2>
inline __device__ void AdamInternal(size_t gid,
                                    T1* param_in,
                                    T2 grad,
                                    T1* exp_avg_in,
                                    T1* exp_avg_sq_in,
                                    T1* max_exp_avg_sq_in,
                                    double lr,
                                    double beta1,
                                    double beta2,
                                    double weight_decay,
                                    double eps,
                                    int step,
                                    bool amsgrad,
                                    bool maximize,
                                    T1* param_out,
                                    half* param_out_fp16,
                                    T1* exp_avg_out,
                                    T1* exp_avg_sq_out,
                                    T1* max_exp_avg_sq_out)
{
    T1 param      = param_in[gid];
    T1 exp_avg    = exp_avg_in[gid];
    T1 exp_avg_sq = exp_avg_sq_in[gid];

    float bias_correction1 = 1 - pow(beta1, step);
    float bias_correction2 = 1 - pow(beta2, step);

    if(maximize)
        grad *= -1;
    if(weight_decay != 0)
        grad += param * weight_decay;

    exp_avg    = exp_avg * beta1 + grad * (1 - beta1);
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);

    T1 denom;
    if(amsgrad)
    {
        T1 max_exp_avg_sq = max_exp_avg_sq_in[gid];
        if(exp_avg_sq > max_exp_avg_sq)
        {
            max_exp_avg_sq          = exp_avg_sq;
            max_exp_avg_sq_out[gid] = max_exp_avg_sq;
        }

        denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
    }
    else
    {
        denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
    }

    float step_size = lr / bias_correction1;
    param           = param - step_size * exp_avg / denom;

    if(param_out_fp16)
        param_out_fp16[gid] = (half)param;
    param_out[gid]      = param;
    exp_avg_out[gid]    = exp_avg;
    exp_avg_sq_out[gid] = exp_avg_sq;
}

extern "C" __global__ void AdamPacked(PARAM_TYPE* params_in,
                                      PARAM_TYPE* grad_in,
                                      PARAM_TYPE* exp_avg_in,
                                      PARAM_TYPE* exp_avg_sq_in,
                                      PARAM_TYPE* max_exp_avg_sq_in,
                                      int step,
                                      double lr,
                                      double beta1,
                                      double beta2,
                                      double weight_decay,
                                      double eps,
                                      bool amsgrad,
                                      bool maximize,
                                      PARAM_TYPE* param_out,
                                      PARAM_TYPE* exp_avg_out,
                                      PARAM_TYPE* exp_avg_sq_out,
                                      PARAM_TYPE* max_exp_avg_sq_out,
                                      long input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= input_size)
        return;

    AdamInternal<PARAM_TYPE, PARAM_TYPE>(gid,
                                         params_in,
                                         grad_in[gid],
                                         exp_avg_in,
                                         exp_avg_sq_in,
                                         max_exp_avg_sq_in,
                                         lr,
                                         beta1,
                                         beta2,
                                         weight_decay,
                                         eps,
                                         step,
                                         amsgrad,
                                         maximize,
                                         param_out,
                                         nullptr,
                                         exp_avg_out,
                                         exp_avg_sq_out,
                                         max_exp_avg_sq_out);
}

extern "C" __global__ void AmpAdamPacked(PARAM_TYPE* param_in,
                                         GRAD_TYPE* grad_in,
                                         PARAM_TYPE* exp_avg_in,
                                         PARAM_TYPE* exp_avg_sq_in,
                                         PARAM_TYPE* max_exp_avg_sq_in,
                                         int32_t* grad_scale,
                                         bool* found_inf,
                                         int* step,
                                         double lr,
                                         double beta1,
                                         double beta2,
                                         double weight_decay,
                                         double eps,
                                         bool amsgrad,
                                         bool maximize,
                                         PARAM_TYPE* param_out,
                                         half* param_out_fp16,
                                         PARAM_TYPE* exp_avg_out,
                                         PARAM_TYPE* exp_avg_sq_out,
                                         PARAM_TYPE* max_exp_avg_sq_out,
                                         long input_size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.x;

    if(gid >= input_size)
        return;

    __shared__ int step_val;
    __shared__ bool skip;
    __shared__ float scale_factor;

    if(lid == 0)
    {
        skip         = (found_inf) ? *found_inf : false;
        scale_factor = (grad_scale) ? *grad_scale : 1.0f;
        step_val     = *step + 1;
    }
    __syncthreads();

    if(skip)
        return;

    PARAM_TYPE grad = grad_in[gid] / scale_factor;

    AdamInternal<PARAM_TYPE, GRAD_TYPE>(gid,
                                        param_in,
                                        grad,
                                        exp_avg_in,
                                        exp_avg_sq_in,
                                        max_exp_avg_sq_in,
                                        lr,
                                        beta1,
                                        beta2,
                                        weight_decay,
                                        eps,
                                        step_val,
                                        amsgrad,
                                        maximize,
                                        param_out,
                                        param_out_fp16,
                                        exp_avg_out,
                                        exp_avg_sq_out,
                                        max_exp_avg_sq_out);
}

extern "C" __global__ void AdamUpdateStep(bool* found_inf, int* step)
{
    if(found_inf && *found_inf)
        return;

    *step += 1;
}
