#include "include/datasets/computer_vision/image_classification/celeba.h"

namespace xt::datasets
{
    // ---------------------- CelebA ---------------------- //

    CelebA::CelebA(const std::string& root): CelebA::CelebA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode): CelebA::CelebA(
        root, mode, false, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CelebA::CelebA(
            root, mode, download, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer) : CelebA::CelebA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        this->root = fs::path(root);
        this->dataset_path = fs::path(root) / this->dataset_folder_name;
        cout << this->dataset_path << endl;
        fs::create_directories(this->dataset_path);
        check_resources();
        load_data();
    }


    void CelebA::load_data()
    {
        // cout << this->dataset_path << endl;
        fs::path pth = this->root / dataset_folder_name / images_folder;
        // cout << pth << endl;

        for (auto& file : fs::directory_iterator(pth))
        {
            if (!file.is_directory())
            {
                files.push_back(file.path());
                // torch::Tensor tensor = xt::utils::image::convertImageToTensor(file.path());
                // data.push_back(tensor);
                targets.push_back(0);
            }
        }
    }

    void CelebA::check_resources()
    {
        fs::path pth = this->root / dataset_folder_name / images_folder;

        bool should_download = false;
        if (fs::exists(pth))
        {
            int cnt = xt::utils::fs::countFiles(pth);
            if (cnt < 202'599)
            {
                fs::remove_all(pth);
                should_download = true;
            }
        }
        if (should_download)
        {
            for (auto resource : resources)
            {
                auto [file_id , md5_hash , file_name] = resource;
                fs::path abs_path = this->dataset_path / file_name;
                xt::utils::download_from_google_drive(file_id, md5_hash, abs_path.string());
            }
        }
    }


    torch::data::Example<> CelebA::get(size_t index)
    {
        torch::Tensor tensor = xt::utils::image::convertImageToTensor(files[index]);

        if (transformer != nullptr)
        {
            tensor = std::any_cast<torch::Tensor>((*transformer)({tensor}));
        }

        return {tensor, torch::tensor(targets[index])};
    }

    torch::optional<size_t> CelebA::size() const
    {
        return files.size();
    }
}
