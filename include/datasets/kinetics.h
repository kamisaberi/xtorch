#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Kinetics : torch::data::Dataset<Kinetics> {

    public :
       Kinetics();
    };
}
