#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"
#include "../../utils/filesystem.h"
#include "../../utils/string.h"
#include <fstream>
#include <map>

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
     * @class Food101
     * @brief Implementation of the Food-101 dataset loader
     * @author Kamran Saberifard
     *
     * Provides loading and access to the Food-101 dataset:
     * - 101,000 food images across 101 categories
     * - 1,000 images per category (750 train + 250 test)
     * - Automatic download and verification
     * - Class name to index mapping
     * - Inherits all transform capabilities from BaseDataset
     */
    class Food101 : public BaseDataset {
    public:
        /**
         * @brief Construct Food101 dataset with root directory only
         * @param root Root directory where dataset will be stored/located
         * @author Kamran Saberifard
         */
        explicit Food101(const std::string &root);

        /**
         * @brief Construct Food101 dataset with specified mode
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @author Kamran Saberifard
         */
        Food101(const std::string &root, DataMode mode);

        /**
         * @brief Construct Food101 dataset with download capability
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @author Kamran Saberifard
         */
        Food101(const std::string &root, DataMode mode, bool download);

        /**
         * @brief Construct Food101 dataset with transforms
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @param transforms Sequence of tensor transformations to apply
         * @author Kamran Saberifard
         */
        Food101(const std::string &root, DataMode mode, bool download,
               vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        /// @brief Official Food-101 dataset download URL
        /// @author Kamran Saberifard
        std::string url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";

        /// @brief Expected filename for downloaded archive
        /// @author Kamran Saberifard
        fs::path dataset_file_name = "food-101.tar.gz";

        /// @brief MD5 checksum for archive verification
        /// @author Kamran Saberifard
        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";

        /// @brief Name of folder containing extracted dataset
        /// @author Kamran Saberifard
        fs::path dataset_folder_name = "food-101";

        /// @brief Total number of images in dataset (101 classes × 1000 images)
        /// @author Kamran Saberifard
        std::size_t images_number = 101'000;

        /// @brief Vector of all 101 food class names
        /// @author Kamran Saberifard
        std::vector<string> classes_name;

        /// @brief Mapping from class names to numerical indices
        /// @author Kamran Saberifard
        std::map<string, int> classes_map;

        /**
         * @brief Loads image data and labels from disk
         * @author Kamran Saberifard
         *
         * @details Reads image files and creates corresponding tensors,
         * populating the data and labels vectors
         */
        void load_data();

        /**
         * @brief Loads class names and creates mapping
         * @author Kamran Saberifard
         *
         * @details Reads class names from meta files and creates
         * both the classes_name vector and classes_map dictionary
         */
        void load_classes();

        /**
         * @brief Verifies dataset files and downloads if needed
         * @author Kamran Saberifard
         *
         * @details Checks for existing files, verifies integrity,
         * and triggers download if files are missing or corrupted
         */
        void check_resources();
    };
}