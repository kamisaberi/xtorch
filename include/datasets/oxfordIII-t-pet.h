#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class OxfordIIITPet : torch::data::Dataset<OxfordIIITPet> {

    public :
       OxfordIIITPet();
    };
}
