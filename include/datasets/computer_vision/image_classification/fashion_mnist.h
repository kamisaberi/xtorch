#pragma once
#include "include/datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class FashionMNIST : public xt::datasets::Dataset
    {
    public:


        explicit FashionMNIST(const std::string& root);
        FashionMNIST(const std::string& root, xt::datasets::DataMode mode);
        FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer);
        FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer,
              std::unique_ptr<xt::Module> target_transformer);

    private:
        std::string url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        fs::path dataset_folder_name = "FashionMNIST/raw";

        vector<tuple<fs::path, std::string>> resources = {
            {fs::path("train-images-idx3-ubyte.gz"), "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
            {fs::path("train-labels-idx1-ubyte.gz"), "25c81989df183df01b3e8a0aad5dffbe"},
            {fs::path("t10k-images-idx3-ubyte.gz"), "bef4ecab320f06d8554ea6380940ec79"},
            {fs::path("t10k-labels-idx1-ubyte.gz"), "bb300cfdad3c16e7a12a480ee83cd310"},
        };

        std::map<std::string, std::tuple<fs::path, fs::path>> files = {
            {"train", {fs::path("train-images-idx3-ubyte"), fs::path("train-labels-idx1-ubyte")}},
            {"test", {fs::path("t10k-images-idx3-ubyte"), fs::path("t10k-labels-idx1-ubyte")}}
        };

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);


        void load_data();

        void check_resources();
    };
}
