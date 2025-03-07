#pragma once

//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <map>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"
#include "../definitions/transforms.h"
#include "../types/arguments.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    // std::vector<torch::Tensor> read_mnist_images(const std::string &file_path, int num_images);

    // std::vector<uint8_t> read_mnist_labels(const std::string &file_path, int num_labels);

    class MNISTBase : public torch::data::Dataset<MNISTBase> {
    public:
        MNISTBase(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        MNISTBase(const fs::path &root, DatasetArguments args);

        void read_images(const std::string &file_path, int num_images);

        void read_labels(const std::string &file_path, int num_labels);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    protected:
        std::vector<torch::Tensor> data; // Store image data as tensors
        std::vector<uint8_t> labels; // Store labels
        DataMode mode = DataMode::TRAIN;
        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void transform_data(std::vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);

    private:
        vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {};
    };


    class MNIST : public MNISTBase {
    public :
        MNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        MNIST(const fs::path &root, DatasetArguments args);

    private :
        std::string url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        fs::path dataset_folder_name = "MNIST/raw";

        vector<tuple<fs::path, std::string> > resources = {
            {fs::path("train-images-idx3-ubyte.gz"), "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
            {fs::path("train-labels-idx1-ubyte.gz"), "d53e105ee54ea40749a09fcbcd1e9432"},
            {fs::path("t10k-images-idx3-ubyte.gz"), "9fb629c4189551a2d022fa330f9573f3"},
            {fs::path("t10k-labels-idx1-ubyte.gz"), "ec29112dd5afa0611ce80d1b7f02629c"},
        };

        std::map<std::string, std::tuple<fs::path, fs::path> > files = {
            {"train", {fs::path("train-images-idx3-ubyte"), fs::path("train-labels-idx1-ubyte")}},
            {"test", {fs::path("t10k-images-idx3-ubyte"), fs::path("t10k-labels-idx1-ubyte")}}
        };

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };


    class FashionMNIST : MNISTBase {
    public:
        FashionMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FashionMNIST(const fs::path &root, DatasetArguments args);

    private:
        std::string url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
        fs::path dataset_folder_name = "FashionMNIST/raw";

        vector<tuple<fs::path, std::string> > resources = {
            {fs::path("train-images-idx3-ubyte.gz"), "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
            {fs::path("train-labels-idx1-ubyte.gz"), "25c81989df183df01b3e8a0aad5dffbe"},
            {fs::path("t10k-images-idx3-ubyte.gz"), "bef4ecab320f06d8554ea6380940ec79"},
            {fs::path("t10k-labels-idx1-ubyte.gz"), "bb300cfdad3c16e7a12a480ee83cd310"},
        };

        std::map<std::string, std::tuple<fs::path, fs::path> > files = {
            {"train", {fs::path("train-images-idx3-ubyte"), fs::path("train-labels-idx1-ubyte")}},
            {"test", {fs::path("t10k-images-idx3-ubyte"), fs::path("t10k-labels-idx1-ubyte")}}
        };

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };


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

    class EMNIST : public MNISTBase {
    public :
        EMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);
        EMNIST(const fs::path &root, DatasetArguments args);

    private:
        std::string url = "https://biometrics.nist.gov/cs_links/EMNIST/";
        fs::path dataset_folder_name = "EMNIST/raw";
        fs::path archive_file_name = "gzip.zip";
        std::string archive_file_md5 = "58c8d27c78d21e728a6bc7b3cc06412e";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);

    };

    class QMNIST : public torch::data::Dataset<QMNIST> {
    private:
        std::vector<torch::Tensor> data; // Store image data as tensors
        std::vector<uint8_t> labels; // Store labels
        torch::Tensor image_;
        torch::Tensor label_;
        std::string url = "https://raw.githubusercontent.com/facebookresearch/qmnist/master/";
        fs::path root;
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

    public :
        QMNIST(const std::string &root, bool train = true, bool download = false);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

        void load_data(bool train);
    };
}
