

# File tensor-dataset.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**tensor-dataset.h**](tensor-dataset_8h.md)

[Go to the documentation of this file](tensor-dataset_8h.md)


```C++
#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class TensorDataset : public BaseDataset {
    public :
        TensorDataset(const std::string &file_path);
        TensorDataset(const std::string &file_path,DataMode mode);
        TensorDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };





}
```


