#pragma once


#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class CarlaStereo : torch::data::Dataset<CarlaStereo> {

    public :
       CarlaStereo();
    };
   class CREStereo : torch::data::Dataset<CREStereo> {

    public :
       CREStereo();
    };
}
