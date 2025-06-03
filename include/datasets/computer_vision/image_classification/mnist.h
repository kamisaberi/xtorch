#pragma once

#include "include/datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class MNIST : public xt::datasets::Dataset
    {
    public :
        explicit MNIST(const std::string& root);
        MNIST(const std::string& root, xt::datasets::DataMode mode);
        MNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        MNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer);
        MNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer,
              std::unique_ptr<xt::Module> target_transformer);


        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;


    private :
        std::string url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        fs::path dataset_folder_name = "MNIST/raw";

        vector<tuple<fs::path, std::string>> resources = {
            {fs::path("train-images-idx3-ubyte.gz"), "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
            {fs::path("train-labels-idx1-ubyte.gz"), "d53e105ee54ea40749a09fcbcd1e9432"},
            {fs::path("t10k-images-idx3-ubyte.gz"), "9fb629c4189551a2d022fa330f9573f3"},
            {fs::path("t10k-labels-idx1-ubyte.gz"), "ec29112dd5afa0611ce80d1b7f02629c"},
        };

        std::map<std::string, std::tuple<fs::path, fs::path>> files = {
            {"train", {fs::path("train-images-idx3-ubyte"), fs::path("train-labels-idx1-ubyte")}},
            {"test", {fs::path("t10k-images-idx3-ubyte"), fs::path("t10k-labels-idx1-ubyte")}}
        };

        void load_data();

        void check_resources();
        void transform_data();


        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);

        bool download = false;
        fs::path root;
        fs::path dataset_path;
    };
}
