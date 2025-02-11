#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Kitti : torch::data::Dataset<Kitti> {

    public :
       Kitti();
    };
   class KittiFlow : torch::data::Dataset<KittiFlow> {

    public :
       KittiFlow();
    };
}
