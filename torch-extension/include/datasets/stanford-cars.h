#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class StanfordCars : torch::data::Dataset<StanfordCars> {

    public :
       StanfordCars();
    };
}
