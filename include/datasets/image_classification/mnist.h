#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"
#include "datasets/base/mnist_base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets
{
    class MNIST : public MNISTBase
    {
    public :
        explicit MNIST(const std::string& root);
        MNIST(const std::string& root, DataMode mode);
        MNIST(const std::string& root, DataMode mode, bool download);
        MNIST(const std::string& root, DataMode mode, bool download,
              vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

        // // MNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);
        // MNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false , std::shared_ptr<xt::data::transforms::Compose> compose= nullptr);
        // MNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false, vector<std::function<torch::Tensor(torch::Tensor)>> transforms= {});

        MNIST(const fs::path& root, DatasetArguments args);

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

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string& root, bool download = false);
        void transform_data();
    };
}
