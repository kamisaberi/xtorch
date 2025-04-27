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
namespace xt::data::datasets {
    /**
     * @enum ImageType
     * @brief Enumeration of available image resolution types for Imagenette dataset
     * @author Kamran Saberifard
     */
    enum class ImageType {
        FULL = 0,  ///< Full resolution images
        PX320 = 1, ///< 320px resolution images
        PX160 = 2  ///< 160px resolution images
    };

    /**
     * @var ImageTypeToString
     * @brief Mapping between ImageType enum and string representations
     * @author Kamran Saberifard
     */
    const std::unordered_map<ImageType, std::string> ImageTypeToString = {
        {ImageType::FULL, "full"},
        {ImageType::PX320, "320px"},
        {ImageType::PX160, "160px"}
    };

    /**
     * @brief Convert ImageType enum to its string representation
     * @param type The ImageType enum value to convert
     * @return std::string String representation of the image type
     * @author Kamran Saberifard
     */
    inline std::string getImageTypeValue(ImageType type) {
        auto it = ImageTypeToString.find(type);
        if (it != ImageTypeToString.end()) {
            return it->second;
        }
        return "unknown"; // Default value if enum not found
    }

    /**
     * @class Imagenette
     * @brief Implementation of the Imagenette dataset loader
     * @author Kamran Saberifard
     *
     * Provides loading and access to the Imagenette dataset:
     * - Subset of 10 classes from ImageNet
     * - Multiple resolution options (full, 320px, 160px)
     * - Automatic download and verification
     * - Train/test split handling
     * - Inherits all transform capabilities from BaseDataset
     */
    class Imagenette : public BaseDataset {
    public:
        /**
         * @brief Construct Imagenette dataset with root directory only
         * @param root Root directory where dataset will be stored/located
         * @author Kamran Saberifard
         */
        explicit Imagenette(const std::string &root);

        /**
         * @brief Construct Imagenette dataset with specified mode
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @author Kamran Saberifard
         */
        Imagenette(const std::string &root, DataMode mode);

        /**
         * @brief Construct Imagenette dataset with download capability
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @author Kamran Saberifard
         */
        Imagenette(const std::string &root, DataMode mode, bool download);

        /**
         * @brief Construct Imagenette dataset with specific image type
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @param type Image resolution type (FULL, PX320, PX160)
         * @author Kamran Saberifard
         */
        Imagenette(const std::string &root, DataMode mode, bool download, ImageType type);

        /**
         * @brief Construct Imagenette dataset with transforms
         * @param root Root directory where dataset will be stored/located
         * @param mode Dataset mode (TRAIN or TEST)
         * @param download Whether to automatically download if missing
         * @param type Image resolution type (FULL, PX320, PX160)
         * @param transforms Sequence of tensor transformations to apply
         * @author Kamran Saberifard
         */
        Imagenette(const std::string &root, DataMode mode, bool download,
                  ImageType type, TransformType transforms);

    private:
        /**
         * @brief Resource configuration for different image types
         * @author Kamran Saberifard
         *
         * @details Contains tuples of:
         * - Download URL
         * - Archive filename
         * - Extracted folder name
         * - MD5 checksum
         */
        std::map<string, std::tuple<fs::path, fs::path, fs::path, std::string>> resources = {
            {
                "full", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"),
                    fs::path("imagenette2.tgz"),
                    fs::path("imagenette2"),
                    "fe2fc210e6bb7c5664d602c3cd71e612"
                }
            },
            {
                "320px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"),
                    fs::path("imagenette2-320.tgz"),
                    fs::path("imagenette2-320"),
                    "3df6f0d01a2c9592104656642f5e78a3"
                }
            },
            {
                "160px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"),
                    fs::path("imagenette2-160.tgz"),
                    fs::path("imagenette2-160"),
                    "e793b78cc4c9e9a4ccc0c1155377a412"
                }
            }
        };

        ImageType type = ImageType::PX160;  ///< Default image resolution type
        fs::path dataset_folder_name = "imagenette";  ///< Base folder name for dataset
        vector<string> labels_name;  ///< List of class label names

        /**
         * @brief Load image data and labels from disk
         * @param mode Dataset mode (TRAIN or TEST)
         * @author Kamran Saberifard
         */
        void load_data(DataMode mode = DataMode::TRAIN);

        /**
         * @brief Verify and download dataset resources if needed
         * @param root Root directory path
         * @param download Whether to download if resources missing
         * @author Kamran Saberifard
         */
        void check_resources(const std::string &root, bool download = false);
    };
}