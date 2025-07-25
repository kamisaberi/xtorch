#include <datasets/computer_vision/3d_point_cloud_analysis/model_net40.h>

namespace xt::datasets
{
    // ---------------------- ModelNet40 ---------------------- //

    ModelNet40::ModelNet40(const std::string& root): ModelNet40::ModelNet40(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ModelNet40::ModelNet40(const std::string& root, xt::datasets::DataMode mode): ModelNet40::ModelNet40(
        root, mode, false, nullptr, nullptr)
    {
    }

    ModelNet40::ModelNet40(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ModelNet40::ModelNet40(
            root, mode, download, nullptr, nullptr)
    {
    }

    ModelNet40::ModelNet40(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ModelNet40::ModelNet40(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ModelNet40::ModelNet40(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ModelNet40::load_data()
    {

    }

    void ModelNet40::check_resources()
    {

    }
}
