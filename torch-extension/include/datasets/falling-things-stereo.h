#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class FallingThingsStereo : torch::data::Dataset<FallingThingsStereo> {

    public :
       FallingThingsStereo();
    };
}
