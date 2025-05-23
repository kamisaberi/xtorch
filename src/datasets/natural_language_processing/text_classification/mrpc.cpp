#include "datasets/natural_language_processing/text_classification/mrpc.h"


namespace xt::data::datasets
{
    // ---------------------- MRPC ---------------------- //

    MRPC::MRPC(const std::string& root): MRPC::MRPC(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MRPC::MRPC(const std::string& root, xt::datasets::DataMode mode): MRPC::MRPC(
        root, mode, false, nullptr, nullptr)
    {
    }

    MRPC::MRPC(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MRPC::MRPC(
            root, mode, download, nullptr, nullptr)
    {
    }

    MRPC::MRPC(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MRPC::MRPC(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MRPC::MRPC(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MRPC::load_data()
    {

    }

    void MRPC::check_resources()
    {

    }
}
