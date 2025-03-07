#pragma once
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
   class StanfordCars : torch::data::Dataset<StanfordCars> {

    public :
       StanfordCars();
    };
}
