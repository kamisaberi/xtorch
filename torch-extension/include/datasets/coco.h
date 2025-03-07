#pragma once
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
   class CocoDetection : torch::data::Dataset<CocoDetection> {

    public :
       CocoDetection();
    };
   class CocoCaptions : torch::data::Dataset<CocoCaptions> {

    public :
       CocoCaptions();
    };
}
