#pragma once
#include "base.h"
#include "../base/datasets.h"
#include  "../utils/filesystem.h"
#include  "../utils/string.h"
#include <fstream>
#include <map>


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
        std::size_t images_number = 101'000;
        std::vector<string> classes_name ;
        std::map<string , int> classes_map ;
        void load_data();
        void load_classes();

        void check_resources();
    };
}
