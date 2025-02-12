#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class UCF101 : torch::data::Dataset<UCF101> {

    public :
       UCF101();
    };
}
