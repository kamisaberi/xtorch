#pragma once

#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class FakeData : torch::data::Dataset<FakeData> {

    public :
       FakeData();
    };
}
