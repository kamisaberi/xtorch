#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class SBDataset : torch::data::Dataset<SBDataset> {

    public :
       SBDataset();
    };
}
