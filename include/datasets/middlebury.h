#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Middlebury2014Stereo : torch::data::Dataset<Middlebury2014Stereo> {

    public :
       Middlebury2014Stereo();
    };
}
