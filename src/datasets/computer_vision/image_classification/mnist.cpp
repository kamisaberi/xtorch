#include <datasets/computer_vision/image_classification/mnist.h>

namespace xt::datasets
{
    // torch::data::Example<> MNIST::get(size_t index) {
    //     cout << "MNIST GET" << endl;
    //     return {data[index], torch::tensor(targets[index])};
    // }
    //
    // torch::optional <size_t> MNIST::size() const {
    //     cout << "MNIST SIZE" << endl;
    //     return data.size();
    // }


    MNIST::MNIST(const std::string& root): MNIST(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MNIST::MNIST(const std::string& root, xt::datasets::DataMode mode): MNIST(
        root, mode, false, nullptr, nullptr)
    {
    }

    MNIST::MNIST(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MNIST(root, mode, download, nullptr, nullptr)
    {
    }

    MNIST::MNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                 std::unique_ptr<xt::Module> transformer) : MNIST(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MNIST::MNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                 std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer)), root(root),
        download(download)
    {
        check_resources();
        load_data();
    }


    void MNIST::check_resources()
    {
        this->root = fs::path(root);
        if (!fs::exists(this->root))
        {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path))
        {
            fs::create_directories(this->dataset_path);
        }

        bool res = true;
        for (const auto& resource : this->resources)
        {
            fs::path pth = std::get<0>(resource);
            std::string md = std::get<1>(resource);
            fs::path fpth = this->dataset_path / pth;
            if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md))
            {
                if (download)
                {
                    string u = (this->url / pth).string();
                    auto [r, path] = xt::utils::download(u, this->dataset_path.string());
                }
                else
                {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            xt::utils::extractGzip(fpth);
        }
    }


    void MNIST::load_data()
    {
        if (mode == xt::datasets::DataMode::TRAIN)
        {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            this->read_images(imgs.string(), 60000);
            this->read_labels(lbls.string(), 60000);
        }
        else
        {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            this->read_images(imgs.string(), 10000);
            this->read_labels(imgs.string(), 10000);
        }
    }

    void MNIST::transform_data()
    {
        for (int i = 0; i < this->data.size(); i++)
        {
            torch::Tensor tensor = this->data[i];
            tensor = std::any_cast<torch::Tensor>((*transformer)({tensor}));
            this->data[i] = tensor;
        }
    }


    void MNIST::read_images(const std::string& file_path, int num_images)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items, rows, cols;
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_items), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert endianess
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        std::vector<torch::Tensor> fimages;
        std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < num_images; i++)
        {
            file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
            torch::Tensor tensor_image = torch::from_blob(images[i].data(), {1, 28, 28},
                                                          torch::kByte).clone();
            if (transformer != nullptr)
            {
                tensor_image = std::any_cast<torch::Tensor>((*transformer)({tensor_image}));
            }
            fimages.push_back(tensor_image);
        }
        file.close();
        this->data = fimages;
    }

    void MNIST::read_labels(const std::string& file_path, int num_labels)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items;
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_items), 4);

        // Convert endianess
        // cout << magic_number << "\t";
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char*>(labels.data()), num_labels);

        // cout << labels.data() << endl;
        file.close();
        this->targets = labels;
    }
}
