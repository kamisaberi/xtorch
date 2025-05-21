#include "datasets/computer_vision/object_detection/xview.h"

namespace xt::data::datasets
{
    // ---------------------- XView ---------------------- //

    XView::XView(const std::string& root): XView::XView(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    XView::XView(const std::string& root, xt::datasets::DataMode mode): XView::XView(
        root, mode, false, nullptr, nullptr)
    {
    }

    XView::XView(const std::string& root, xt::datasets::DataMode mode, bool download) :
        XView::XView(
            root, mode, download, nullptr, nullptr)
    {
    }

    XView::XView(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : XView::XView(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    XView::XView(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void XView::load_data()
    {

    }

    void XView::check_resources()
    {

    }
}
