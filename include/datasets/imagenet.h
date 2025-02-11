#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class ImageNet : torch::data::Dataset<ImageNet> {

    public :
       ImageNet();
    };
}
