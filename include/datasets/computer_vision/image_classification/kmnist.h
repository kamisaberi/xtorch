#pragma once
#include "../../common.h"



using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets {

    class KMNIST : public xt::datasets::Dataset {
    public :

        explicit KMNIST(const std::string& root);
        KMNIST(const std::string& root, xt::datasets::DataMode mode);
        KMNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        KMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer);
        KMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer,
              std::unique_ptr<xt::Module> target_transformer);


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

        void load_data();

        void check_resources();

        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);

        bool download = false;
        fs::path root;
        fs::path dataset_path;

    };

}
