#pragma once

#include "../base/datasets.h"


namespace torch::ext::data::datasets {
   class Omniglot : torch::data::Dataset<Omniglot> {

    public :
       Omniglot();
    };
}
