#pragma once

#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class VOCSegmentation : torch::data::Dataset<VOCSegmentation> {

    public :
       VOCSegmentation();
    };
   class VOCDetection : torch::data::Dataset<VOCDetection> {

    public :
       VOCDetection();
    };
}
