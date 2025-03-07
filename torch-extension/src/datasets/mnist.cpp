#include "../../include/datasets/mnist.h"

namespace torch::ext::data::datasets {
    // std::vector<torch::Tensor> read_mnist_images(const std::string &file_path, int num_images) {
    //     std::ifstream file(file_path, std::ios::binary);
    //     if (!file.is_open()) {
    //         throw std::runtime_error("Failed to open file: " + file_path);
    //     }
    //
    //     // Read metadata
    //     int32_t magic_number, num_items, rows, cols;
    //     file.read(reinterpret_cast<char *>(&magic_number), 4);
    //     file.read(reinterpret_cast<char *>(&num_items), 4);
    //     file.read(reinterpret_cast<char *>(&rows), 4);
    //     file.read(reinterpret_cast<char *>(&cols), 4);
    //
    //     // Convert endianess
    //     magic_number = __builtin_bswap32(magic_number);
    //     num_items = __builtin_bswap32(num_items);
    //     rows = __builtin_bswap32(rows);
    //     cols = __builtin_bswap32(cols);
    //
    //     std::vector<torch::Tensor> fimages;
    //     std::vector<std::vector<uint8_t> > images(num_images, std::vector<uint8_t>(rows * cols));
    //     for (int i = 0; i < num_images; i++) {
    //         file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
    //         auto tensor_image = torch::from_blob(images[i].data(), {1, 28, 28},
    //                                              torch::kByte).clone();
    //         fimages.push_back(tensor_image);
    //     }
    //     //        cout << fimages[0] << endl;
    //
    //
    //     file.close();
    //     return fimages;
    // }
    //
    // std::vector<uint8_t> read_mnist_labels(const std::string &file_path, int num_labels) {
    //     std::ifstream file(file_path, std::ios::binary);
    //     if (!file.is_open()) {
    //         throw std::runtime_error("Failed to open file: " + file_path);
    //     }
    //
    //     // Read metadata
    //     int32_t magic_number, num_items;
    //     file.read(reinterpret_cast<char *>(&magic_number), 4);
    //     file.read(reinterpret_cast<char *>(&num_items), 4);
    //
    //     // Convert endianess
    //     cout << magic_number << "\t";
    //     magic_number = __builtin_bswap32(magic_number);
    //     num_items = __builtin_bswap32(num_items);
    //
    //     std::vector<uint8_t> labels(num_labels);
    //     file.read(reinterpret_cast<char *>(labels.data()), num_labels);
    //
    //     // cout << labels.size() << endl;
    //     file.close();
    //     return labels;
    // }

    MNISTBase::MNISTBase(const std::string &root, DataMode mode, bool download) {
        this->root = root;
        this->download = download;
        this->mode = mode;
        // check_resources(root, download);
        // load_data(mode);
    }

    MNISTBase::MNISTBase(const fs::path &root, DatasetArguments args) {
        auto [mode , download , transforms] = args;
        this->root = root;
        this->download = download;
        this->mode = mode;
        this->transforms = transforms;
        // check_resources(root, download);
        // load_data(mode);
    }


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
            auto tensor_image = torch::from_blob(images[i].data(), {1, 28, 28},
                                                 torch::kByte).clone();
            fimages.push_back(tensor_image);
        }
        //        cout << fimages[0] << endl;


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
        cout << magic_number << "\t";
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), num_labels);

        cout << labels.data() << endl;
        file.close();
        this->labels = labels;
    }

    void MNISTBase::transform_data(vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms) {
        std::vector<torch::Tensor> data; // Store image data as tensors
        // std::vector<uint8_t> targets; // Store image data as tensors
        // cout << "transforms.size:" << transforms.size() << endl;
        for (const auto &transform: transforms) {
            // std::cout << "1" << std::endl;
            auto data_tensor = this->map(transform).map(torch::data::transforms::Stack<>());
            // std::cout << "2" << " " << this->data.size() << "  " << data_tensor.size().value() << std::endl;
            auto data_loader = torch::data::make_data_loader(std::move(data_tensor), /*batch_size=*/this->data.size());
            // std::cout << "3" << std::endl;
            for (auto &batch: *data_loader) {
                // std::cout << "3 " << i << " " << batch.data.sizes() << std::endl;
                data = batch.data.unbind(0);
                // targets.push_back(batch.data()->target[0].to(torch::kUInt8)) ;
            }
            // std::cout << "4" << std::endl;
            // torch::Tensor full_data = torch::cat(data, 0);
            // torch::Tensor full_targets = torch::cat(targets, 0);
            this->data = data;
            // std::cout << "5  " << this->data.size() << std::endl;
            // this->labels = targets;
        }

        // std::cout << "10" << std::endl;
        auto dt = this->map(torch::data::transforms::Stack<>());
        // std::cout << "11" << std::endl;
        auto data_loader = torch::data::make_data_loader(std::move(dt), /*batch_size=*/this->data.size());
        // std::cout << "12" << std::endl;
        for (auto &batch: *data_loader) {
            data = batch.data.unbind(0);
        }
        // std::cout << "13" << std::endl;
        this->data = data;
        // std::cout << "14  " << this->data.size() << std::endl;
    }

    torch::data::Example<> MNISTBase::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> MNISTBase::size() const {
        return data.size();
    }


    //------------------ MNIST ------------------//
    MNIST::MNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        // cout << "MNIST SIZE: " << this->labels.size() << endl;
        check_resources(root, download);
        load_data(mode);
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        // cout << "MNIST SIZE: " << this->labels.size() << endl;
        // cout << this->data[0].sizes() << endl;
        // cout << this->labels << endl;
    }

    MNIST::MNIST(const fs::path &root, DatasetArguments args) : MNISTBase(root, args) {
        auto [mode , download , transforms] = args;
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        check_resources(root, download);
        load_data(mode);
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        if (!transforms.empty()) {
            // cout << "Transforms 11111111111111111111" << endl;
            this->transform_data(transforms);
        }
        // cout << "MNIST SIZE: " << this->data.size() << endl;
    }

    // torch::data::Example<> MNIST::get(size_t index) {
    //     return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    // }
    //
    // torch::optional<size_t> MNIST::size() const {
    //     return data.size();
    // }

    void MNIST::check_resources(const std::string &root, bool download) {
        this->root = fs::path(root);
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        bool res = true;
        for (const auto &resource: this->resources) {
            fs::path pth = std::get<0>(resource);
            std::string md = std::get<1>(resource);
            fs::path fpth = this->dataset_path / pth;
            if (!(fs::exists(fpth) && torch::ext::utils::get_md5_checksum(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = torch::ext::utils::download_data(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            torch::ext::utils::extractGzip(fpth);
        }
    }


    void MNIST::load_data(DataMode mode) {
        if (mode == DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs.string() << "  " << lbls.string() << endl;
            cout << imgs << endl;
            this->read_images(imgs.string(), 50000);
            this->read_labels(lbls.string(), 50000);
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << labels[0] << endl;
            // this->data = images;
            // this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(imgs.string(), 10000);
            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // this->data = images;
            // this->labels = labels;
        }
    }

    // void MNIST::transform_data(vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms) {
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

    FashionMNIST::FashionMNIST(const std::string &root, DataMode mode,
                               bool download) : MNISTBase(root, mode, download) {
        check_resources(root, download);
        load_data(mode);
    }

    void FashionMNIST::load_data(DataMode mode) {
        if (mode == DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs.string() << "  " << lbls.string() << endl;
            cout << imgs << endl;
            this->read_images(imgs.string(), 50000);
            this->read_labels(lbls.string(), 50000);
            //
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << labels[0] << endl;
            // this->data = images;
            // this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(lbls.string(), 10000);
            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // this->data = images;
            // this->labels = labels;
        }
    }

    // void FashionMNIST::load_data(bool train) {
    //     if (train) {
    //         fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
    //         fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
    //         cout << imgs << endl;
    //         auto images = read_mnist_images(imgs.string(), 50000);
    //         auto labels = read_mnist_labels(lbls.string(), 50000);
    //         cout << images.size() << endl;
    //         cout << labels.size() << endl;
    //         this->data = images;
    //         this->labels = labels;
    //         //             for (int i = 0; i < 100; i++) {
    //         // //                for (auto row : images[i]) {
    //         // //                    cout << (unsigned int) row << " -- ";
    //         // //                }
    //         //                 cout << (unsigned int) labels[i] << "\t";
    //         //             }
    //         //             cout << endl;
    //     } else {
    //         fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
    //         fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
    //         cout << imgs << endl;
    //         auto images = read_mnist_images(imgs.string(), 10000);
    //         auto labels = read_mnist_labels(lbls.string(), 10000);
    //         cout << images.size() << endl;
    //         cout << labels.size() << endl;
    //         this->data = images;
    //         this->labels = labels;
    //     }
    // }

    void FashionMNIST::check_resources(const std::string &root, bool download) {
        this->root = fs::path(root);
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        bool res = true;
        for (const auto &resource: this->resources) {
            fs::path pth = std::get<0>(resource);
            std::string md = std::get<1>(resource);
            fs::path fpth = this->dataset_path / pth;
            if (!(fs::exists(fpth) && torch::ext::utils::get_md5_checksum(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = torch::ext::utils::download_data(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            torch::ext::utils::extractGzip(fpth);
        }
    }

    KMNIST::KMNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
        check_resources(root, download);
        load_data(mode);
    }


    void KMNIST::load_data(DataMode mode) {
        if (mode == DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs.string() << "  " << lbls.string() << endl;
            cout << imgs << endl;
            this->read_images(imgs.string(), 50000);
            this->read_labels(lbls.string(), 50000);
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << labels[0] << endl;
            // this->data = images;
            // this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(lbls.string(), 10000);
            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // this->data = images;
            // this->labels = labels;
        }
    }


    EMNIST::EMNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
        check_resources(root, download);
        load_data(mode);
    }


    void EMNIST::load_data(DataMode mode) {
        // if (mode == DataMode::TRAIN) {
        //     fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
        //     fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
        //     cout << imgs.string() << "  " << lbls.string() << endl;
        //     cout << imgs << endl;
        //     auto images = read_mnist_images(imgs.string(), 50000);
        //     auto labels = read_mnist_labels(lbls.string(), 50000);
        //     cout << labels[0] << endl;
        //     this->data = images;
        //     this->labels = labels;
        // } else {
        //     fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
        //     fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
        //     cout << imgs << endl;
        //     auto images = read_mnist_images(imgs.string(), 10000);
        //     auto labels = read_mnist_labels(lbls.string(), 10000);
        //     this->data = images;
        //     this->labels = labels;
        // }
    }


    QMNIST::QMNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        // cout << "MNIST SIZE: " << this->labels.size() << endl;
        check_resources(root, download);
        load_data(mode);
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        // cout << "MNIST SIZE: " << this->labels.size() << endl;
        // cout << this->data[0].sizes() << endl;
        // cout << this->labels << endl;
    }

    QMNIST::QMNIST(const fs::path &root, DatasetArguments args) : MNISTBase(root, args) {
        auto [mode , download , transforms] = args;
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        check_resources(root, download);
        load_data(mode);
        // cout << "MNIST SIZE: " << this->data.size() << endl;
        if (!transforms.empty()) {
            // cout << "Transforms 11111111111111111111" << endl;
            this->transform_data(transforms);
        }
        // cout << "MNIST SIZE: " << this->data.size() << endl;
    }


    void QMNIST::load_data(DataMode mode ) {
        if (mode == DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(resources["train"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["train"][1]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 50000);
            this->read_labels(lbls.string(), 50000);
            //
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << images.size() << endl;
            // cout << labels.size() << endl;
            // this->data = images;
            // this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(resources["test"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["test"][1]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(lbls.string(), 10000);

            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // cout << images.size() << endl;
            // cout << labels.size() << endl;
            // this->data = images;
            // this->labels = labels;
        }
    }
}
