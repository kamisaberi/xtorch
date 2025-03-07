#pragma once
#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class Kinetics : torch::data::Dataset<Kinetics> {

    public :
       Kinetics();
    };
}
