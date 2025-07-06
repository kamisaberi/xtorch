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
        this->dataset_path = fs::path(root) / this->dataset_folder_name;
        check_resources();
        load_data();
    }


    void CelebA::load_data()
    {
        fs::path pth = this->root / dataset_folder_name / images_folder;

        for (auto& file : fs::directory_iterator(pth))
        {
            if (!file.is_directory())
            {
                files.push_back(file.path());
                // torch::Tensor tensor = xt::utils::image::convertImageToTensor(file.path());
                // data.push_back(tensor);
                // targets.push_back(labels_name.size() - 1);
            }
        }


    }

    void CelebA::check_resources()
    {
    }


    torch::data::Example<> CelebA::get(size_t index)
    {
        torch::Tensor tensor = xt::utils::image::convertImageToTensor(files[index]);

        return {tensor, torch::tensor(targets[index])};
    }

    torch::optional<size_t> CelebA::size() const
    {
        return files.size();
    }

}
