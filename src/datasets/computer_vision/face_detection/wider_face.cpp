#include "datasets/computer_vision/face_detection/wider_face.h"

namespace xt::data::datasets {

    // ---------------------- Caltech101 ---------------------- //

    WIDERFace::WIDERFace(const std::string& root): WIDERFace::WIDERFace(
            root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WIDERFace::WIDERFace(const std::string& root, xt::datasets::DataMode mode): WIDERFace::WIDERFace(
            root, mode, false, nullptr, nullptr)
    {
    }

    WIDERFace::WIDERFace(const std::string& root, xt::datasets::DataMode mode, bool download) :
            WIDERFace::WIDERFace(
                    root, mode, download, nullptr, nullptr)
    {
    }

    WIDERFace::WIDERFace(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WIDERFace::WIDERFace(
            root, mode, download, std::move(transformer), nullptr)
    {
    }

    WIDERFace::WIDERFace(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
            xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Caltech101::load_data()
    {

    }

    void Caltech101::check_resources()
    {

    }


}
