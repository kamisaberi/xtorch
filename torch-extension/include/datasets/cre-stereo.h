#pragma once


#include "../base/datasets.h"
#include "base.h"



namespace torch::ext::data::datasets {
   class CarlaStereo : BaseDataset {

    public :
       CarlaStereo();
    };
   class CREStereo : torch::data::Dataset<CREStereo> {

    public :
       CREStereo();
    };
}
