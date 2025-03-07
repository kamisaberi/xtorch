#pragma once

#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class FlyingChairs : torch::data::Dataset<FlyingChairs> {

    public :
       FlyingChairs();
    };
   class FlyingThings3D : torch::data::Dataset<FlyingThings3D> {

    public :
       FlyingThings3D();
    };
}
