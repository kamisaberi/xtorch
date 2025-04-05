
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



    struct RandomFlip {
    private:
        double horizontal_prob;
        double vertical_prob;

    public:
        RandomFlip(double h_prob = 0.5, double v_prob = 0.0);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };



}