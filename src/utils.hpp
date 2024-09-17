#pragma once

#include "pico-gpt2.hpp"

#include <vector>

#include "imgui.h"

inline std::pair<std::vector<double>, std::vector<double>>
generate_gelu_points(double start, double stop, double step)
{
    std::vector<double> x_points;
    std::vector<double> y_points;

    x_points.reserve((size_t)((stop - start) / step));
    y_points.reserve((size_t)((stop - start) / step));

    for (auto x = start; x <= stop; x += step)
    {
        x_points.push_back(x);
        y_points.push_back(pico_gpt2::gelu(x));
    }
 
    return {x_points, y_points};
}

inline std::vector<double> 
generate_linear_gradient(double start, double stop, double step)
{
    std::vector<double> input_data;
    input_data.reserve((size_t)((stop - start) / step));

    for (auto x = start; x <= stop; x += step)
    {
        input_data.push_back(x);
    }

    return input_data;
}

inline std::pair<std::vector<double>, std::vector<double>>
generate_softmax_points(const std::vector<double> &input)
{
    std::vector<double> x_points(input.size());
    std::vector<double> y_points = pico_gpt2::softmax(input);

    // Fill x_points with indices
    std::iota(x_points.begin(), x_points.end(), 0);

    return {x_points, y_points};
}

inline std::pair<std::vector<double>, std::vector<double>> 
generate_layer_norm_points(const std::vector<double> &x, 
                           const std::vector<double> &gamma, 
                           const std::vector<double> &beta) 
{
    std::vector<double> y = pico_gpt2::layer_norm(x, gamma, beta);

    return {x, y};
}
