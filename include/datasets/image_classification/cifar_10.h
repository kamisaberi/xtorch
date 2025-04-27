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
     * @class CIFAR10
     * @brief Implementation of the CIFAR-10 dataset loader
     * @author Kamran Saberifard
     *
     * Provides loading and access to the CIFAR-10 dataset:
     * - 60,000 32x32 color images
     * - 10 classes (airplane, automobile, bird, etc.)
     * - Automatic download and verification
     * - Train/test split handling
     * - Inherits all transform capabilities from BaseDataset
     */
    class CIFAR10 : public BaseDataset
    {
    public:
        /**
         * @brief Construct CIFAR10 dataset with root directory only
         * @param root Root directory where dataset will be stored/located
         * @author Kamran Saberifard
         */
        explicit CIFAR10(const std::string& root);

        /**
         * @brief Construct CIFAR10 dataset with specified mode
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @author Kamran Saberifard
         */
        CIFAR10(const std::string& root, DataMode mode);

        /**
         * @brief Construct CIFAR10 dataset with download capability
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @author Kamran Saberifard
         */
        CIFAR10(const std::string& root, DataMode mode, bool download);

        /**
         * @brief Construct CIFAR10 dataset with transforms
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @param transforms Sequence of tensor transformations to apply
         * @author Kamran Saberifard
         */
        CIFAR10(const std::string& root, DataMode mode, bool download, TransformType transforms);

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
        /// @brief Official CIFAR-10 dataset download URL
        /// @author Kamran Saberifard
        std::string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

        /// @brief Expected filename for downloaded archive
        /// @author Kamran Saberifard
        fs::path archive_file_name = "cifar-10-binary.tar.gz";

        /// @brief MD5 checksum for archive verification
        /// @author Kamran Saberifard
        std::string archive_file_md5 = "c32a1d4ab5d03f1284b67883e8d87530";

        /// @brief Name of folder containing extracted dataset
        /// @author Kamran Saberifard
        fs::path dataset_folder_name = "cifar-10-batches-bin";

        /// @brief List of training set binary files
        /// @author Kamran Saberifard
        vector<fs::path> train_file_names = {
            fs::path("data_batch_1.bin"),
            fs::path("data_batch_2.bin"),
            fs::path("data_batch_3.bin"),
            fs::path("data_batch_4.bin"),
            fs::path("data_batch_5.bin")
        };

        /// @brief Test set binary filename
        /// @author Kamran Saberifard
        fs::path test_file_name = "test_batch.bin";

        /**
         * @brief Internal method to load data from binary files
         * @param mode Whether to load training or test data
         * @author Kamran Saberifard
         */
        void load_data(DataMode mode = DataMode::TRAIN);
    };
}