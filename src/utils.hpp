#pragma once

#include "pico-gpt2.hpp"

#include <atomic>
#include <vector>

#include "imgui.h"

struct shared_state
{
    std::atomic<bool> running;
};

std::vector<float>
generate_gelu_points(float start, float stop, float step)
{
    std::vector<float> points;
    points.reserve((size_t)((stop - start) / step));

    for (float x = start; x <= stop; x += step)
    {
        points.push_back((float) pico_gpt2::gelu(x));
    }

    return points;
}

