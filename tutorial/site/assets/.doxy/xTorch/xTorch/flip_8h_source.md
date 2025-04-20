

# File flip.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**flip.h**](flip_8h.md)

[Go to the documentation of this file](flip_8h.md)


```C++

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
```


