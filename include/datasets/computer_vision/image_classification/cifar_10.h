#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class CIFAR10 : public xt::datasets::Dataset
    {
    public:
        explicit CIFAR10(const std::string& root);

        CIFAR10(const std::string& root, xt::datasets::DataMode mode);
        CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download);
        CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer);

        CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer,
                std::unique_ptr<xt::Module> target_transformer);


        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

        fs::path archive_file_name = "cifar-10-binary.tar.gz";

        std::string archive_file_md5 = "c32a1d4ab5d03f1284b67883e8d87530";

        fs::path dataset_folder_name = "cifar-10-batches-bin";

        vector<fs::path> train_file_names = {
            fs::path("data_batch_1.bin"),
            fs::path("data_batch_2.bin"),
            fs::path("data_batch_3.bin"),
            fs::path("data_batch_4.bin"),
            fs::path("data_batch_5.bin")
        };

        fs::path test_file_name = "test_batch.bin";


        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();
    };
}
