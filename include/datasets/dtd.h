#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class DTD : torch::data::Dataset<DTD> {

    public :
       DTD();
    };
}
