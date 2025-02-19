#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
    class LFW : public torch::data::Dataset<LFW> {
    private :

    };

    class LFWPeople : public LFW {

    public :
        LFWPeople();
    };

    class LFWPairs : public LFW {

    public :
        LFWPairs();
    };
}
