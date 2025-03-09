#pragma once
#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
   class Cityscapes : BaseDataset<Cityscapes> {

    public :
       Cityscapes();
    };
}
