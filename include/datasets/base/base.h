#pragma once

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
namespace xt::data::datasets {

    /**
     * @class BaseDataset
     * @brief Abstract base class for all custom datasets in the xt framework
     *
     * Inherits from torch::data::Dataset to provide standard PyTorch dataset
     * functionality while adding common features needed across xt datasets.
     *
     * Features:
     * - Built-in transform support via Compose
     * - Standardized data/labels storage
     * - Download capability
     * - Mode handling (train/test/val)
     *
     * @note Derived classes must implement get() and size() methods
     */
    class BaseDataset : public torch::data::Dataset<BaseDataset> {

    public:
        /// Type alias for transform sequences (vector of tensor-to-tensor functions)
        using TransformType = vector<std::function<torch::Tensor(torch::Tensor)> >;

        /**
         * @brief Construct a BaseDataset with root directory only
         * @param root Root directory where dataset is stored
         */
        explicit BaseDataset(const std::string &root);

        /**
         * @brief Construct a BaseDataset with root and mode
         * @param root Root directory where dataset is stored
         * @param mode Dataset mode (TRAIN/TEST/VAL)
         */
        BaseDataset(const std::string &root, DataMode mode);

        /**
         * @brief Construct a BaseDataset with download capability
         * @param root Root directory where dataset is stored
         * @param mode Dataset mode (TRAIN/TEST/VAL)
         * @param download Whether to download dataset if not present
         */
        BaseDataset(const std::string &root, DataMode mode , bool download);

        /**
         * @brief Construct a BaseDataset with transforms
         * @param root Root directory where dataset is stored
         * @param mode Dataset mode (TRAIN/TEST/VAL)
         * @param download Whether to download dataset if not present
         * @param transforms Sequence of tensor transforms to apply
         */
        BaseDataset(const std::string &root, DataMode mode , bool download ,
                  vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

        /**
         * @brief Get a dataset example by index
         * @param index Index of example to retrieve
         * @return torch::data::Example<> containing data and label
         *
         * @note Pure virtual - must be implemented by derived classes
         */
        torch::data::Example<> get(size_t index) override;

        /**
         * @brief Get the size of the dataset
         * @return torch::optional<size_t> Number of examples in dataset
         *
         * @note Pure virtual - must be implemented by derived classes
         */
        torch::optional<size_t> size() const override;

    public:
        /// Storage for image/feature data as tensors
        std::vector<torch::Tensor> data;

        /// Storage for labels as uint8 values
        std::vector<uint8_t> labels;

        /// Current dataset mode (TRAIN/TEST/VAL)
        DataMode mode = DataMode::TRAIN;

        /// Download flag indicating if dataset should download when missing
        bool download = false;

        /// Filesystem path to dataset root directory
        fs::path root;

        /// Filesystem path to specific dataset location
        fs::path dataset_path;

        /// Transform composition pipeline
        xt::data::transforms::Compose compose;

        /// Sequence of tensor transforms to apply
        vector<std::function<torch::Tensor(torch::Tensor)>> transforms = {};

    private:
        // Potential future extension point for example-level transforms
        // vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {};
    };
}