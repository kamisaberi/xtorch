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
   class Kitti2012Stereo : torch::data::Dataset<Kitti2012Stereo> {

    public :
       Kitti2012Stereo();
    };
   class Kitti2015Stereo : torch::data::Dataset<Kitti2015Stereo> {

    public :
       Kitti2015Stereo();
    };
}
