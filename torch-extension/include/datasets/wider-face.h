#pragma once
#include "../headers/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class WIDERFace : BaseDataset {
    public :
        explicit WIDERFace(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        WIDERFace(const fs::path &root, DatasetArguments args);

    private:
        fs::path dataset_folder_name = fs::path("widerface");
        std::vector<std::tuple<std::string, std::string, fs::path> > resources = {
                {"15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M", "3fedf70df600953d25982bcd13d91ba2", fs::path("WIDER_train.zip")},
                {"1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q", "dfa7d7e790efa35df3788964cf0bbaea", fs::path("WIDER_val.zip")},
                {"1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T", "e5d8f4248ed24c334bbd12f49c29dd40", fs::path("WIDER_test.zip")}
        };
        std::tuple<fs::path, std::string, fs::path> ANNOTATIONS_FILE = {
                fs::path("http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"),
                "0e3767bcf0e326556d407bf5bff5d27c",
                fs::path("wider_face_split.zip")
        };
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);


    };
}
