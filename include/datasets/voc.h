#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class VOCSegmentation : torch::data::Dataset<VOCSegmentation> {

    public :
       VOCSegmentation();
    };
   class VOCDetection : torch::data::Dataset<VOCDetection> {

    public :
       VOCDetection();
    };
}
