#pragma once
#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class CIFAR100 : public xt::datasets::Dataset
    {
    public:
        explicit CIFAR100(const std::string& root);

        CIFAR100(const std::string& root, xt::datasets::DataMode mode);

        CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download);

        CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download,
        std::unique_ptr<xt::Module> transformer);

        CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer,
                std::unique_ptr<xt::Module> target_transformer);




        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";

        fs::path archive_file_name = "cifar-100-binary.tar.gz";

        std::string archive_file_md5 = "03b5dce01913d631647c71ecec9e9cb8";

        fs::path dataset_folder_name = "cifar-100-binary";

        fs::path train_file_name = "train.bin";

        fs::path test_file_name = "test.bin";


        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();
    };
}