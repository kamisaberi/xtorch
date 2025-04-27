#include "../../../include/datasets/image_classification/cifar_10.h"

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
    /* ------------------ CIFAR10 Implementation ------------------ */

    /**
     * @brief Default constructor with root directory only
     * @param root Root directory path for dataset storage
     * @author Kamran Saberifard
     *
     * @note Delegates to main constructor with:
     * - Mode: TRAIN
     * - Download: false
     */
    CIFAR10::CIFAR10(const std::string& root): CIFAR10::CIFAR10(root, DataMode::TRAIN, false)
    {
        // Simple delegation to main constructor with default params
    }

    /**
     * @brief Constructor with root directory and mode
     * @param root Root directory path for dataset storage
     * @param mode Dataset mode (TRAIN/TEST)
     * @author Kamran Saberifard
     *
     * @note Delegates to main constructor with download=false
     */
    CIFAR10::CIFAR10(const std::string& root, DataMode mode): CIFAR10::CIFAR10(root, mode, false)
    {
        // Delegation to handle mode while keeping download disabled
    }

    /**
     * @brief Main constructor with download capability
     * @param root Root directory path for dataset storage
     * @param mode Dataset mode (TRAIN/TEST)
     * @param download Whether to download dataset if missing
     * @author Kamran Saberifard
     *
     * @details Handles:
     * - Filesystem path initialization
     * - Optional downloading with checksum verification
     * - Archive extraction
     * - Data loading
     */
    CIFAR10::CIFAR10(const std::string& root, DataMode mode, bool download) : BaseDataset(root, mode, download)
    {
        // Initialize filesystem paths for dataset access
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;

        bool res = true;  // Track success of download/extraction operations

        // Handle automatic download if requested
        if (download)
        {
            bool should_download = false;

            // Check if archive file exists
            if (!fs::exists(this->root / this->archive_file_name))
            {
                should_download = true;  // Download if file missing
            }
            else
            {
                // Verify existing file integrity with MD5 checksum
                std::string md5 = xt::utils::get_md5_checksum((this->root / this->archive_file_name).string());
                if (md5 != archive_file_md5)
                {
                    should_download = true;  // Download if checksum mismatch
                }
            }

            // Execute download if required
            if (should_download)
            {
                auto [result, path] = xt::utils::download(this->url, this->root.string());
                res = result;  // Store download success status
            }

            // Extract archive if download succeeded
            if (res)
            {
                string pth = (this->root / this->archive_file_name).string();
                res = xt::utils::extract(pth, this->root);
            }
        }

        // Load data from files (uses existing files if download failed)
        load_data(mode);
    }

    /**
     * @brief Constructor with transforms support
     * @param root Root directory path for dataset storage
     * @param mode Dataset mode (TRAIN/TEST)
     * @param download Whether to download dataset if missing
     * @param transforms Sequence of tensor transformations
     * @author Kamran Saberifard
     *
     * @note Maintains same functionality as main constructor while
     * adding transform support through base class initialization
     */
    CIFAR10::CIFAR10(const std::string& root, DataMode mode, bool download,
                     TransformType transforms) : BaseDataset(root, mode, download, transforms)
    {
        // Same initialization as main constructor
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;
        bool res = true;

        // Download handling (identical to main constructor)
        if (download)
        {
            bool should_download = false;
            if (!fs::exists(this->root / this->archive_file_name))
            {
                should_download = true;
            }
            else
            {
                std::string md5 = xt::utils::get_md5_checksum((this->root / this->archive_file_name).string());
                if (md5 != archive_file_md5)
                {
                    should_download = true;
                }
            }
            if (should_download)
            {
                auto [result, path] = xt::utils::download(this->url, this->root.string());
                res = result;
            }
            if (res)
            {
                string pth = (this->root / this->archive_file_name).string();
                res = xt::utils::extract(pth, this->root);
            }
        }

        // Load data (transforms will be applied through base class)
        load_data(mode);
    }

    /**
     * @brief Retrieve a single sample from the dataset
     * @param index Position of sample to retrieve
     * @return torch::data::Example<> containing image tensor and label
     * @author Kamran Saberifard
     *
     * @note Returns cloned tensor to ensure memory safety and proper
     * ownership management in the PyTorch ecosystem
     */
    torch::data::Example<> CIFAR10::get(size_t index)
    {
        // Clone both data and label tensors to prevent memory issues
        return {data[index].clone(), torch::tensor(labels[index])};
    }

    /**
     * @brief Get the total number of samples in dataset
     * @return torch::optional<size_t> Dataset size
     * @author Kamran Saberifard
     *
     * @note Returns actual size of loaded data (typically 50,000 for train)
     */
    torch::optional<size_t> CIFAR10::size() const
    {
        return data.size();  // Return count of loaded samples
    }

    /**
     * @brief Internal method to load CIFAR-10 binary data
     * @param mode Whether to load training or test data
     * @author Kamran Saberifard
     *
     * @details Handles:
     * - Binary file parsing (1 label byte + 3072 image bytes per sample)
     * - Label extraction and type conversion
     * - Image tensor conversion and reshaping
     * - Channel permutation to PyTorch standard (C, H, W)
     * - Data storage in member vectors
     */
    void CIFAR10::load_data(DataMode mode)
    {
        const int num_files = 5;  // CIFAR-10 has 5 training batch files

        // Process each training file in sequence
        for (auto path : this->train_file_names)
        {
            // Construct full path to binary file
            std::string file_path = this->dataset_path / path;

            // Open file in binary mode
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;  // Skip to next file if open fails
            }

            // Each file contains exactly 10,000 samples
            for (int j = 0; j < 10000; ++j)
            {
                // Read single byte label (0-9)
                uint8_t label;
                file.read(reinterpret_cast<char*>(&label), sizeof(label));
                labels.push_back(static_cast<int64_t>(label));  // Store as int64

                // Read image data (32x32x3 = 3072 bytes)
                std::vector<uint8_t> image(3072);
                file.read(reinterpret_cast<char*>(image.data()), image.size());

                // Convert to tensor and reshape to 3x32x32
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                   torch::kByte).clone();  // Clone for ownership

                // Permute dimensions from (C, H, W) to (C, W, H) and back to (C, H, W)
                // (Note: Original CIFAR-10 binary format has unusual dimension ordering)
                tensor_image = tensor_image.permute({0, 2, 1});

                // Store final tensor in data vector
                data.push_back(tensor_image);
            }

            file.close();  // Explicit close (RAII would handle this but clear intent)
        }
    }
}