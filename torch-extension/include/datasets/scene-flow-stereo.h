#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class SceneFlowStereo : torch::data::Dataset<SceneFlowStereo> {

    public :
       SceneFlowStereo();
    };
}
