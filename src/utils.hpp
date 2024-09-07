#pragma once

#include "pico-gpt2.hpp"

#include <atomic>
#include <vector>

#include "imgui.h"

struct shared_state
{
    std::atomic<bool> running;
};

std::vector<double>
generate_gelu_points(double start, double stop, double step)
{
    std::vector<double> points;
    points.reserve((size_t)((stop - start) / step));

    for (auto x = start; x <= stop; x += step)
    {
        points.push_back(pico_gpt2::gelu(x));
    }

    return points;
}
