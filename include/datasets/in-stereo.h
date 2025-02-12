#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class InStereo2k : torch::data::Dataset<InStereo2k> {

    public :
       InStereo2k();
    };
}
