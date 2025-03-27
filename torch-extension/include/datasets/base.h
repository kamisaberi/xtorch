#pragma once

#include "../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class BaseDataset : public torch::data::Dataset<BaseDataset> {
    public:
        explicit BaseDataset(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        BaseDataset(const fs::path &root, DatasetArguments args);

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

        void transform_data(std::vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);

    private:
        vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {};
    };
}
