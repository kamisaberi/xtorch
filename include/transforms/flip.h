
#include "../headers/transforms.h"

namespace xt::data::transforms {

    struct HorizontalFlip {
    public:
        HorizontalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };


    struct VerticalFlip {
    public:
        VerticalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };




}