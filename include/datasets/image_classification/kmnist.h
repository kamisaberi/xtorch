#pragma once
#include "../../headers/datasets.h"
#include "../../datasets/base/mnist_base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {

    class KMNIST : public MNISTBase {
    public :
        KMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        KMNIST(const fs::path &root, DatasetArguments args);

    private:
        std::string url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/";
        fs::path dataset_folder_name = "KMNIST/raw";

        vector<tuple<fs::path, std::string> > resources = {
            {fs::path("train-images-idx3-ubyte.gz"), "bdb82020997e1d708af4cf47b453dcf7"},
            {fs::path("train-labels-idx1-ubyte.gz"), "e144d726b3acfaa3e44228e80efcd344"},
            {fs::path("t10k-images-idx3-ubyte.gz"), "5c965bf0a639b31b8f53240b1b52f4d7"},
            {fs::path("t10k-labels-idx1-ubyte.gz"), "7320c461ea6c1c855c0b718fb2a4b134"},
        };

        std::map<std::string, std::tuple<fs::path, fs::path> > files = {
            {"train", {fs::path("train-images-idx3-ubyte"), fs::path("train-labels-idx1-ubyte")}},
            {"test", {fs::path("t10k-images-idx3-ubyte"), fs::path("t10k-labels-idx1-ubyte")}}
        };

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

}
