

# File base.h

[**File List**](files.md) **>** [**base**](dir_9657a40feeddef1c88c156c7806ef467.md) **>** [**base.h**](datasets_2base_2base_8h.md)

[Go to the documentation of this file](datasets_2base_2base_8h.md)


```C++
#pragma once

#include "../../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class BaseDataset : public torch::data::Dataset<BaseDataset> {

    public:
        using TransformType = vector<std::function<torch::Tensor(torch::Tensor)> >;

        BaseDataset(const std::string &root);
        BaseDataset(const std::string &root, DataMode mode);
        BaseDataset(const std::string &root, DataMode mode , bool download);
        BaseDataset(const std::string &root, DataMode mode , bool download , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


            // virtual ~BaseDataset() = 0;

        // BaseDataset(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    public:
        std::vector<torch::Tensor> data; // Store image data as tensors
        std::vector<uint8_t> labels; // Store labels
        DataMode mode = DataMode::TRAIN;
        bool download = false;
        fs::path root;
        fs::path dataset_path;
        xt::data::transforms::Compose compose;
        vector<std::function<torch::Tensor(torch::Tensor)>> transforms = {};


    private:
        // vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {};

    };
}
```


