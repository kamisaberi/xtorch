#pragma once
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
   class Sintel : torch::data::Dataset<Sintel> {

    public :
       Sintel();
    };
   class SintelStereo : torch::data::Dataset<SintelStereo> {

    public :
       SintelStereo();
    };
}
