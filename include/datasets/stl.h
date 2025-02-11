#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class STL10 : torch::data::Dataset<STL10> {

    public :
       STL10();
    };
}
