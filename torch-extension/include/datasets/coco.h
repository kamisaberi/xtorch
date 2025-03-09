#pragma once
#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
   class CocoDetection : BaseDataset{

    public :
       CocoDetection();
    };
   class CocoCaptions : BaseDataset{

    public :
       CocoCaptions();
    };
}
