

# File rotation.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**rotation.h**](rotation_8h.md)

[Go to the documentation of this file](rotation_8h.md)


```C++

#include "../headers/transforms.h"

namespace xt::data::transforms {

    struct Rotation {
    public:
        Rotation(double angle_deg);

        torch::Tensor operator()(const torch::Tensor &input_tensor);

        private :
            double angle; // Rotation angle in degrees
    };


    // struct Rotation {
    // public:
    //     Rotation(float angle);
    //
    //     torch::Tensor operator()(torch::Tensor input);
    //
    // private:
    //     float angle;
    // };



}
```


