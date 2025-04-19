#include "../../../include/datasets/text-classification/ag-news.h"

namespace xt::data::datasets {

    AgNews::AgNews(const std::string &root): AgNews::AgNews(root, DataMode::TRAIN, false) {
    }

    AgNews::AgNews(const std::string &root, DataMode mode): AgNews::AgNews(root, mode, false) {
    }

    AgNews::AgNews(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("AgNews: AgNews not implemented");
    }


    AgNews::AgNews(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("AgNews: AgNews not implemented");
    }


}
