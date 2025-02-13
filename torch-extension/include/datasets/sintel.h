#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Sintel : torch::data::Dataset<Sintel> {

    public :
       Sintel();
    };
   class SintelStereo : torch::data::Dataset<SintelStereo> {

    public :
       SintelStereo();
    };
}
