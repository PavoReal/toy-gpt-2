#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace pico_gpt2
{
    //
    // GELU
    // https://arxiv.org/pdf/1606.08415
    //
    inline double 
    gelu(double x)
    {
        return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
    }

    //
    // Softmax
    // 
    //
    inline std::vector<double>
    softmax(const std::vector<double> &x)
    {
        // Find the maximum value in x
        double max_val = *std::max_element(x.begin(), x.end());

        // Compute exp(x - max_val) for each element
        std::vector<double> exp_x;
        exp_x.reserve(x.size());

        for (const auto &val : x) 
        {
            exp_x.push_back(std::exp(val - max_val));
        }

        // Compute the sum of exp_x
        double sum_exp_x = std::accumulate(exp_x.begin(), exp_x.end(), 0.0);

        // Normalize exp_x by dividing each element by sum_exp_x
        for (auto &val : exp_x) 
        {
            val /= sum_exp_x;
        }

        return exp_x;
    }
}
