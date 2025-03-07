#pragma once

#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class MovingMNIST : torch::data::Dataset<MovingMNIST> {

    public :
       MovingMNIST();
    };
}
