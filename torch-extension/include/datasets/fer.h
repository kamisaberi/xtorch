#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class FER2013 : torch::data::Dataset<FER2013> {

    public :
       FER2013();
    };
}
