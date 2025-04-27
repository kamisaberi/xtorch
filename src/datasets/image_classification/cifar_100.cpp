#include "../../../include/datasets/image_classification/cifar_100.h"

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
    /* ------------------ CIFAR100 Implementation ------------------ */

    /**
     * @brief Default constructor with root directory only
     * @param root Root directory path for dataset storage
     * @author Kamran Saberifard
     *
     * @note Delegates to main constructor with:
     * - Mode: TRAIN
     * - Download: false
     */
    CIFAR100::CIFAR100(const std::string& root): CIFAR100::CIFAR100(root, DataMode::TRAIN, false)
    {
        /* Constructor chaining to initialize with default parameters */
    }

    /**
     * @brief Constructor with root directory and mode
     * @param root Root directory path for dataset storage
     * @param mode Dataset mode (TRAIN/TEST)
     * @author Kamran Saberifard
     *
     * @note Delegates to main constructor with download=false
     */
    CIFAR100::CIFAR100(const std::string& root, DataMode mode): CIFAR100::CIFAR100(root, mode, false)
    {
        /* Constructor chaining to initialize with specified mode */
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
     *
     * @throws std::runtime_error If download fails or verification fails
     */
    CIFAR100::CIFAR100(const std::string& root, DataMode mode, bool download) : BaseDataset(root, mode, download)
    {
        // Initialize filesystem paths for dataset access
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;

        bool res = true;  // Track download/extraction success

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
    CIFAR100::CIFAR100(const std::string& root, DataMode mode, bool download,
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
     * @param index Position of sample to retrieve (0 <= index < size())
     * @return torch::data::Example<> containing image tensor and label
     * @author Kamran Saberifard
     *
     * @note Returns cloned tensor to ensure memory safety and proper
     * ownership management in the PyTorch ecosystem
     */
    torch::data::Example<> CIFAR100::get(size_t index)
    {
        // Clone both data and label tensors to prevent memory issues
        return {data[index].clone(), torch::tensor(labels[index])};
    }

    /**
     * @brief Get the total number of samples in dataset
     * @return torch::optional<size_t> Dataset size (50,000 train or 10,000 test)
     * @author Kamran Saberifard
     */
    torch::optional<size_t> CIFAR100::size() const
    {
        // Return count of loaded samples
        return data.size();
    }

    /**
     * @brief Internal method to load CIFAR-100 binary data
     * @param mode Whether to load training or test data
     * @author Kamran Saberifard
     *
     * @details Handles:
     * - Binary file parsing (2 label bytes + 3072 image bytes per sample)
     * - Fine label extraction (skips coarse label)
     * - Image tensor conversion and reshaping
     * - Channel permutation to PyTorch standard (C, H, W)
     * - Data storage in member vectors
     */
    void CIFAR100::load_data(DataMode mode)
    {
        // CIFAR-100 has single training file (50000 samples)
        std::string file_path = (dataset_path / train_file_name).string();
        cout << "train file path : " << file_path << endl;

        // Open binary file
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return;
        }

        // Process all 50000 samples in file
        for (int j = 0; j < 50000; ++j)
        {
            // Read labels (CIFAR-100 specific format)
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));  // Skip coarse label (not used)
            file.read(reinterpret_cast<char*>(&label), sizeof(label));  // Read fine label (0-99)

            // Store fine-grained label (100 classes)
            labels.push_back(static_cast<int64_t>(label));

            // Read image data (32x32x3 = 3072 bytes)
            std::vector<uint8_t> image(3072);
            file.read(reinterpret_cast<char*>(image.data()), image.size());

            // Convert to tensor and reshape to 3x32x32
            auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                             torch::kByte).clone();  // Clone for memory ownership

            // Permute dimensions from (C, H, W) to (C, W, H) and back to (C, H, W)
            // (Original CIFAR-100 binary format has unusual dimension ordering)
            tensor_image = tensor_image.permute({0, 2, 1});

            // Store final tensor in data vector
            data.push_back(tensor_image);
        }

        file.close();  // Explicit close (RAII would handle this but clear intent)
    }
}