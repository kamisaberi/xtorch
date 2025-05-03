#pragma once

#include "../../headers/transforms.h"

namespace xt::data::transforms {

    struct Rotation {
    public:
        Rotation(double angle_deg);

        torch::Tensor operator()(const torch::Tensor &input_tensor);

        private :
            double angle; // Rotation angle in degrees
    };


    // struct Rotation {
    // public:
    //     Rotation(float angle);
    //
    //     torch::Tensor operator()(torch::Tensor input);
    //
    // private:
    //     float angle;
    // };



}