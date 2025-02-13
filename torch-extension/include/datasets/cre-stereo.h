#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class CarlaStereo : torch::data::Dataset<CarlaStereo> {

    public :
       CarlaStereo();
    };
   class CREStereo : torch::data::Dataset<CREStereo> {

    public :
       CREStereo();
    };
}
