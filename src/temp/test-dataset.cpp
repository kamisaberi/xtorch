#include "../../include/temp/test-dataset.h"

namespace xt::temp {

    TestDataset::TestDataset(){}
    TestDataset::~TestDataset(){}

    torch::Tensor forward(torch::Tensor input){
        return input;
    }

}