#include "include/datasets/computer_vision/image_captioning/coco_captions.h"

namespace xt::data::datasets {

    // ---------------------- Caltech101 ---------------------- //

    CocoCaptions::CocoCaptions(const std::string& root): CocoCaptions::CocoCaptions(
            root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CocoCaptions::CocoCaptions(const std::string& root, xt::datasets::DataMode mode): CocoCaptions::CocoCaptions(
            root, mode, false, nullptr, nullptr)
    {
    }

    CocoCaptions::CocoCaptions(const std::string& root, xt::datasets::DataMode mode, bool download) :
            CocoCaptions::CocoCaptions(
                    root, mode, download, nullptr, nullptr)
    {
    }

    CocoCaptions::CocoCaptions(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CocoCaptions::CocoCaptions(
            root, mode, download, std::move(transformer), nullptr)
    {
    }

    CocoCaptions::CocoCaptions(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
            xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CocoCaptions::load_data()
    {

    }

    void CocoCaptions::check_resources()
    {

    }


}
