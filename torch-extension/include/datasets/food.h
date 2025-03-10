#pragma once
#include "base.h"
#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
    class Food101 : BaseDataset {
    public :
        Food101(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Food101(const fs::path &root, DatasetArguments args);

    private :
        std::string download_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";
        fs::path archive_file_name = "food-101.tar.gz";
        std::string archive_file_md5 = "85eeb15f3717b99a5da872d97d918f87";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
