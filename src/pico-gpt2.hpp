#pragma once

#include <cmath>

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
}
