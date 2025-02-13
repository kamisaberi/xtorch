#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class INaturalist : torch::data::Dataset<INaturalist> {

    public :
       INaturalist();
    };
}
