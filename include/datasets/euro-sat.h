#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class EuroSAT : torch::data::Dataset<EuroSAT> {

    public :
       EuroSAT();
    };
}
