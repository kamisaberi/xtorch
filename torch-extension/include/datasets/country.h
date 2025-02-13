#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Country211 : torch::data::Dataset<Country211> {

    public :
       Country211();
    };
}
