
#include "../headers/transforms.h"

namespace xt::data::transforms {

    struct Lambda {
    public:
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}