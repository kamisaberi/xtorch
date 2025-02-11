#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class FGVCAircraft : torch::data::Dataset<FGVCAircraft> {

    public :
       FGVCAircraft();
    };
}
