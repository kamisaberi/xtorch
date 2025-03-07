#pragma once

//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <map>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/extract.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"
#include "../definitions/transforms.h"
#include "../types/arguments.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class BaseDataset : public torch::data::Dataset<BaseDataset> {
    public:
        BaseDataset(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        BaseDataset(const fs::path &root, DatasetArguments args);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

        torch::data::datasets::MapDataset<torch::ext::data::datasets::BaseDataset, torch::data::transforms::Stack<>()>
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
