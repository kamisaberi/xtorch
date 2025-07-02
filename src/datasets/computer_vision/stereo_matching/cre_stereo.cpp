#include "include/datasets/computer_vision/stereo_matching/cre_stereo.h"

namespace xt::datasets
{
    // ---------------------- CREStereo ---------------------- //

    CREStereo::CREStereo(const std::string& root): CREStereo::CREStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CREStereo::CREStereo(const std::string& root, xt::datasets::DataMode mode): CREStereo::CREStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    CREStereo::CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CREStereo::CREStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    CREStereo::CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CREStereo::CREStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CREStereo::CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CREStereo::load_data()
    {

    }

    void CREStereo::check_resources()
    {

    }
}
