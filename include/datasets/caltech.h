#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
    class Caltech101 : torch::data::Dataset<Caltech101> {

    public :
        Caltech101();
    };

    class Caltech256 : torch::data::Dataset<Caltech256> {

    public :
        Caltech256();
    };


}

