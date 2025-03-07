#pragma once
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
   class Cityscapes : torch::data::Dataset<Cityscapes> {

    public :
       Cityscapes();
    };
}
