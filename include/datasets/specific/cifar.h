#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class CIFAR10 : public BaseDataset {
    public:
        CIFAR10(const std::string &root);
        CIFAR10(const std::string &root, DataMode mode);
        CIFAR10(const std::string &root, DataMode mode , bool download);
        CIFAR10(const std::string &root, DataMode mode , bool download, TransformType transforms);


        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        // std::vector<torch::Tensor> data; // Store image data as tensors
        // std::vector<int64_t> labels; // Store labels
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
        fs::path archive_file_name = "cifar-10-binary.tar.gz";
        std::string archive_file_md5 = "c32a1d4ab5d03f1284b67883e8d87530";
        // fs::path root;
        // fs::path dataset_path;
        fs::path dataset_folder_name = "cifar-10-batches-bin";
        vector<fs::path> train_file_names = {
            fs::path("data_batch_1.bin"),
            fs::path("data_batch_2.bin"),
            fs::path("data_batch_3.bin"),
            fs::path("data_batch_4.bin"),
            fs::path("data_batch_5.bin")
        };
        fs::path test_file_name = "test_batch.bin";

        void load_data(DataMode mode = DataMode::TRAIN);
    };


    class CIFAR100 : public BaseDataset {
    public:
        CIFAR100(const std::string &root);
        CIFAR100(const std::string &root, DataMode mode);
        CIFAR100(const std::string &root, DataMode mode , bool download);
        CIFAR100(const std::string &root, DataMode mode , bool download, TransformType transforms);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        // std::vector<torch::Tensor> data; // Store image data as tensors
        // std::vector<int64_t> labels; // Store labels
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
        fs::path archive_file_name = "cifar-100-binary.tar.gz";
        std::string archive_file_md5 = "03b5dce01913d631647c71ecec9e9cb8";
        // fs::path root;
        // fs::path dataset_path;
        fs::path dataset_folder_name = "cifar-100-binary";
        fs::path train_file_name = "train.bin";
        fs::path test_file_name = "test.bin";

        void load_data( DataMode mode = DataMode::TRAIN);
    };
}
