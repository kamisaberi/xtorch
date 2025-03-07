#pragma once

#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class FallingThingsStereo : torch::data::Dataset<FallingThingsStereo> {

    public :
       FallingThingsStereo();
    };
}
