#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;

/**
 * @namespace xt::data::datasets
 * @brief Namespace for custom dataset implementations in the xt framework
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 */
namespace xt::data::datasets
{
    /**
     * @class CIFAR100
     * @brief Implementation of the CIFAR-100 dataset loader
     * @author Kamran Saberifard
     *
     * Provides loading and access to the CIFAR-100 dataset:
     * - 60,000 32x32 color images
     * - 100 fine-grained classes
     * - 20 superclasses (coarse labels)
     * - Automatic download and verification
     * - Train/test split handling
     * - Inherits all transform capabilities from BaseDataset
     */
    class CIFAR100 : public BaseDataset
    {
    public:
        /**
         * @brief Construct CIFAR100 dataset with root directory only
         * @param root Root directory where dataset will be stored/located
         * @author Kamran Saberifard
         */
        explicit CIFAR100(const std::string& root);

        /**
         * @brief Construct CIFAR100 dataset with specified mode
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @author Kamran Saberifard
         */
        CIFAR100(const std::string& root, DataMode mode);

        /**
         * @brief Construct CIFAR100 dataset with download capability
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @author Kamran Saberifard
         */
        CIFAR100(const std::string& root, DataMode mode, bool download);

        /**
         * @brief Construct CIFAR100 dataset with transforms
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @param transforms Sequence of tensor transformations to apply
         * @author Kamran Saberifard
         */
        CIFAR100(const std::string& root, DataMode mode, bool download, TransformType transforms);

        /**
         * @brief Get a single sample from the dataset
         * @param index Position of sample to retrieve
         * @return torch::data::Example<> containing image tensor and label
         * @author Kamran Saberifard
         */
        torch::data::Example<> get(size_t index) override;

        /**
         * @brief Get the total number of samples in dataset
         * @return torch::optional<size_t> Dataset size (50,000 train or 10,000 test)
         * @author Kamran Saberifard
         */
        torch::optional<size_t> size() const override;

    private:
        /// @brief Official CIFAR-100 dataset download URL
        /// @author Kamran Saberifard
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";

        /// @brief Expected filename for downloaded archive
        /// @author Kamran Saberifard
        fs::path archive_file_name = "cifar-100-binary.tar.gz";

        /// @brief MD5 checksum for archive verification
        /// @author Kamran Saberifard
        std::string archive_file_md5 = "03b5dce01913d631647c71ecec9e9cb8";

        /// @brief Name of folder containing extracted dataset
        /// @author Kamran Saberifard
        fs::path dataset_folder_name = "cifar-100-binary";

        /// @brief Training set binary filename
        /// @author Kamran Saberifard
        fs::path train_file_name = "train.bin";

        /// @brief Test set binary filename
        /// @author Kamran Saberifard
        fs::path test_file_name = "test.bin";

        /**
         * @brief Internal method to load data from binary files
         * @param mode Whether to load training or test data
         * @author Kamran Saberifard
         */
        void load_data(DataMode mode = DataMode::TRAIN);
    };
}