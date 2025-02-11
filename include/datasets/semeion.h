#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class SEMEION : torch::data::Dataset<SEMEION> {

    public :
       SEMEION();
    };
}
