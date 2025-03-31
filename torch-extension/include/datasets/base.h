#pragma once

#include "../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class BaseDataset : public torch::data::Dataset<BaseDataset> {
    public:

        BaseDataset(const std::string &root);
        BaseDataset(const std::string &root, DataMode mode);
        BaseDataset(const std::string &root, DataMode mode , bool download);
        BaseDataset(const std::string &root, DataMode mode  , bool download, std::shared_ptr<xt::data::transforms::Compose> compose);
        BaseDataset(const std::string &root, DataMode mode , bool download , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


        // BaseDataset(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

        torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset, torch::data::transforms::Stack<>()>
        transform_dataset();

    protected:
        std::vector<torch::Tensor> data; // Store image data as tensors
        std::vector<uint8_t> labels; // Store labels
        DataMode mode = DataMode::TRAIN;
        bool download = false;
        fs::path root;
        fs::path dataset_path;
        xt::data::transforms::Compose compose;


    private:
        // vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {};
        vector<std::function<torch::Tensor(torch::Tensor)>> transforms = {};
    };
}
