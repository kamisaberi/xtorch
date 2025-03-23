#pragma once
#include "base.h"
#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
    class Food101 : public BaseDataset {
    public :
        Food101(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Food101(const fs::path &root, DatasetArguments args);

    private :
        std::string url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";
        fs::path dataset_file_name = "food-101.tar.gz";
        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";
        fs::path dataset_folder_name = "food-101";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
