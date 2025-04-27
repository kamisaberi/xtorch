/**
 * @file imagenette.cpp
 * @brief Implementation of the Imagenette dataset loader
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements the Imagenette dataset loader which provides access to a subset of
 * the ImageNet dataset containing 10 easily classified classes. The implementation supports
 * different image resolutions and data augmentation transforms.
 */

#include "../../../include/datasets/image_classification/imagenette.h"

namespace xt::data::datasets {

    /**
     * @brief Default constructor initializes dataset with default parameters
     * @param root Path to the root directory containing or to contain the dataset
     *
     * Initializes with:
     * - Training mode (DataMode::TRAIN)
     * - Download disabled (false)
     * - Default image size (PX160)
     */
    Imagenette::Imagenette(const std::string &root): Imagenette::Imagenette(
        root, DataMode::TRAIN, false, ImageType::PX160) {
    }

    /**
     * @brief Constructor with specified data mode
     * @param root Path to the root directory
     * @param mode Dataset mode (TRAIN or VALIDATION)
     *
     * Initializes with:
     * - Download disabled (false)
     * - Default image size (PX160)
     */
    Imagenette::Imagenette(const std::string &root, DataMode mode): Imagenette::Imagenette(
        root, mode, false, ImageType::PX160) {
    }

    /**
     * @brief Constructor with download option
     * @param root Path to the root directory
     * @param mode Dataset mode (TRAIN or VALIDATION)
     * @param download Whether to download dataset if not present
     *
     * Initializes with default image size (PX160)
     */
    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download): Imagenette::Imagenette(
        root, mode, download, ImageType::PX160) {
    }

    /**
     * @brief Primary constructor with all basic parameters
     * @param root Path to the root directory
     * @param mode Dataset mode (TRAIN or VALIDATION)
     * @param download Whether to download dataset if not present
     * @param type Image resolution type (PX160, PX320, etc.)
     *
     * Initializes the dataset and loads data according to specified parameters
     */
    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type)
        : BaseDataset(root, mode, download) , type(type) {
        // Verify dataset resources and download if needed
        check_resources(root, download);
        // Load data according to specified mode
        load_data(mode);
    }

    /**
     * @brief Constructor with transforms for data augmentation
     * @param root Path to the root directory
     * @param mode Dataset mode (TRAIN or VALIDATION)
     * @param download Whether to download dataset if not present
     * @param type Image resolution type (PX160, PX320, etc.)
     * @param transforms List of transforms to apply to images
     *
     * Initializes the dataset with specified transforms for data augmentation
     */
    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type,
                           TransformType transforms): BaseDataset(
        root, mode, download) , type(type) {
        // Initialize transforms if provided
        if (!transforms.empty()) {
            this->transforms = transforms;
            this->compose = xt::data::transforms::Compose(transforms);
        }
        // Verify dataset resources
        check_resources(root, download);
        // Load data according to specified mode
        load_data(mode);
    }

    /**
     * @brief Verifies dataset resources and optionally downloads them
     * @param root Path to verify dataset resources
     * @param download Flag to enable automatic download if resources missing
     * @throws std::runtime_error if resources are missing and download is disabled
     *
     * Checks for dataset existence and integrity using MD5 checksums.
     * Downloads and extracts dataset if requested and needed.
     */
    void Imagenette::check_resources(const std::string &root, bool download) {
        // Convert root path to filesystem path
        this->root = fs::path(root);

        // Verify root directory exists
        if (!fs::exists(this->root)) {
            throw runtime_error("Dataset root path does not exist");
        }

        // Set up dataset directory path
        this->dataset_path = this->root / this->dataset_folder_name;

        // Create dataset directory if it doesn't exist
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        // Get resource information based on selected image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];
        fs::path fpth = this->dataset_path / dataset_filename;

        // Verify file exists and has correct checksum
        if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md)) {
            if (download) {
                // Download dataset if enabled
                string u = url.string();
                auto [r, path] = xt::utils::download(u, this->dataset_path.string());
            } else {
                throw runtime_error("Dataset resources not found. Set download=true to download automatically");
            }
        }

        // Extract downloaded archive
        xt::utils::extractTgz(fpth, this->dataset_path.string());
    }

    /**
     * @brief Loads image data and labels based on specified mode
     * @param mode Dataset mode (TRAIN or VALIDATION)
     *
     * Loads images from appropriate subdirectories, converts them to tensors,
     * applies transforms if specified, and stores them with corresponding labels.
     */
    void Imagenette::load_data(DataMode mode) {
        // Get resource information based on image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];

        if (mode == DataMode::TRAIN) {
            // Set path to training data
            fs::path path = this->dataset_path / folder_name / fs::path("train");

            // Iterate through each class directory
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto &img: fs::directory_iterator(p.path())) {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (!transforms.empty()) {
                            tensor = this->compose(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);  // Use label index
                    }
                }
            }
        } else {
            // Set path to validation data
            fs::path path = this->dataset_path / folder_name / fs::path("val");

            // Iterate through each class directory
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto &img: fs::directory_iterator(p.path())) {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (!transforms.empty()) {
                            tensor = this->compose(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);  // Use label index
                    }
                }
            }
        }
    }
} // namespace xt::data::datasets

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Machine Learning Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * Licensed under the MIT License. See LICENSE file in the project root for full license information.
 */