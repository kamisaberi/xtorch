#include "../../../include/datasets/base/mnist_base.h"

namespace xt::data::datasets {


    MNISTBase::MNISTBase(const std::string &root): MNISTBase::MNISTBase(root, DataMode::TRAIN, false) {
    }

    MNISTBase::MNISTBase(const std::string &root, DataMode mode): MNISTBase::MNISTBase(root, mode, false) {
    }

    MNISTBase::MNISTBase(const std::string &root, DataMode mode, bool download) : BaseDataset(root , mode, download) {
    }

    MNISTBase::MNISTBase(const std::string &root, DataMode mode, bool download,
                         vector<std::function<torch::Tensor(torch::Tensor)> > transforms) : MNISTBase::MNISTBase(
        root, mode, download) {
        this->transforms = transforms;
        if (!transforms.empty()) {
            this->compose = xt::data::transforms::Compose(this->transforms);
        }
    }

    // MNISTBase::~MNISTBase() {}

    void MNISTBase::read_images(const std::string &file_path, int num_images) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items, rows, cols;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);
        file.read(reinterpret_cast<char *>(&rows), 4);
        file.read(reinterpret_cast<char *>(&cols), 4);

        // Convert endianess
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        std::vector<torch::Tensor> fimages;
        std::vector<std::vector<uint8_t> > images(num_images, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
            torch::Tensor tensor_image = torch::from_blob(images[i].data(), {1, 28, 28},
                                                          torch::kByte).clone();
            if (!this->transforms.empty()) {
                tensor_image = compose(tensor_image);
            }
            fimages.push_back(tensor_image);
        }

        file.close();
        this->data = fimages;
    }

    void MNISTBase::read_labels(const std::string &file_path, int num_labels) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);

        // Convert endianess
        // cout << magic_number << "\t";
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), num_labels);

        // cout << labels.data() << endl;
        file.close();
        this->labels = labels;
    }

    // void MNISTBase::transform_data(vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms) {
    //     std::vector<torch::Tensor> data; // Store image data as tensors
    //     // std::vector<uint8_t> targets; // Store image data as tensors
    //     // cout << "transforms.size:" << transforms.size() << endl;
    //     for (const auto &transform: transforms) {
    //         // std::cout << "1" << std::endl;
    //         auto data_tensor = this->map(transform).map(torch::data::transforms::Stack<>());
    //         // std::cout << "2" << " " << this->data.size() << "  " << data_tensor.size().value() << std::endl;
    //         auto data_loader = torch::data::make_data_loader(std::move(data_tensor), /*batch_size=*/this->data.size());
    //         // std::cout << "3" << std::endl;
    //         for (auto &batch: *data_loader) {
    //             // std::cout << "3 " << i << " " << batch.data.sizes() << std::endl;
    //             data = batch.data.unbind(0);
    //             // targets.push_back(batch.data()->target[0].to(torch::kUInt8)) ;
    //         }
    //         // std::cout << "4" << std::endl;
    //         // torch::Tensor full_data = torch::cat(data, 0);
    //         // torch::Tensor full_targets = torch::cat(targets, 0);
    //         this->data = data;
    //         // std::cout << "5  " << this->data.size() << std::endl;
    //         // this->labels = targets;
    //     }
    //
    //     // std::cout << "10" << std::endl;
    //     auto dt = this->map(torch::data::transforms::Stack<>());
    //     // std::cout << "11" << std::endl;
    //     auto data_loader = torch::data::make_data_loader(std::move(dt), /*batch_size=*/this->data.size());
    //     // std::cout << "12" << std::endl;
    //     for (auto &batch: *data_loader) {
    //         data = batch.data.unbind(0);
    //     }
    //     // std::cout << "13" << std::endl;
    //     this->data = data;
    //     // std::cout << "14  " << this->data.size() << std::endl;
    // }

    // torch::data::Example<> MNISTBase::get(size_t index) {
    //     return {data[index], torch::tensor(labels[index])};
    // }
    //
    // // Override `size` method to return the number of samples
    // torch::optional<size_t> MNISTBase::size() const {
    //     return data.size();
    // }


}
