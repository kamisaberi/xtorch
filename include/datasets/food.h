#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Food101 : torch::data::Dataset<Food101> {

    public :
       Food101();
    };
}
