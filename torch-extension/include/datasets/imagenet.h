#pragma once
#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class ImageNet : torch::data::Dataset<ImageNet> {

    public :
       ImageNet();
    };
}
