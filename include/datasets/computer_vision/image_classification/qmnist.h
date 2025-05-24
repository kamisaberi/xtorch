#pragma once
// #include "../../../headers/datasets.h"
#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets {


    class QMNIST : public xt::datasets::Dataset {
    public :

        explicit QMNIST(const std::string& root);
        QMNIST(const std::string& root, xt::datasets::DataMode mode);
        QMNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        QMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer);
        QMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer,
              std::unique_ptr<xt::Module> target_transformer);


    private:
        std::string url = "https://raw.githubusercontent.com/facebookresearch/qmnist/master/";
        fs::path dataset_path;
        fs::path dataset_folder_name = "QMNIST/raw";
        std::string archive_file_md5 = "58c8d27c78d21e728a6bc7b3cc06412e";
        std::map<string, std::vector<std::tuple<fs::path, std::string> > > resources = {
            {
                "train", {
                    {fs::path("qmnist-train-images-idx3-ubyte.gz"), "ed72d4157d28c017586c42bc6afe6370"},
                    {fs::path("qmnist-train-labels-idx2-int.gz"), "0058f8dd561b90ffdd0f734c6a30e5e4"}
                }
            },
            {
                "test", {
                    {fs::path("qmnist-test-images-idx3-ubyte.gz"), "1394631089c404de565df7b7aeaf9412"},
                    {fs::path("qmnist-test-labels-idx2-int.gz"), "5b5b05890a5e13444e108efe57b788aa"}
                }
            },
            {
                "nist", {
                    {fs::path("xnist-images-idx3-ubyte.xz"), "7f124b3b8ab81486c9d8c2749c17f834"},
                    {fs::path("xnist-labels-idx2-int.xz"), "5ed0e788978e45d4a8bd4b7caec3d79d"}
                }
            }
        };
        void load_data();

        void check_resources();
        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);

        bool download = false;
        fs::path root;


    };
}
