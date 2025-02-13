#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class FakeData : torch::data::Dataset<FakeData> {

    public :
       FakeData();
    };
}
