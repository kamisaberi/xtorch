#include "../../include/temp/test-dataset.h"

namespace xt::temp {

    TestDataset::TestDataset(){}
//    TestDataset::~TestDataset(){}

    torch::Tensor TestDataset::forward(torch::Tensor input){
        return input;
    }

}