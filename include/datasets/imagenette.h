#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Imagenette : torch::data::Dataset<Imagenette> {

    public :
       Imagenette();
    };
}
