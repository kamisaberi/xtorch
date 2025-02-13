#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class PhotoTour : torch::data::Dataset<PhotoTour> {

    public :
       PhotoTour();
    };
}
