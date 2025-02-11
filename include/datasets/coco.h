#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class CocoDetection : torch::data::Dataset<CocoDetection> {

    public :
       CocoDetection();
    };
   class CocoCaptions : torch::data::Dataset<CocoCaptions> {

    public :
       CocoCaptions();
    };
}
