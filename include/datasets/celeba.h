#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class CelebA : torch::data::Dataset<CelebA> {

    public :
       CelebA();
    };
}
