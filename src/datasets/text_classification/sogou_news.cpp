#include "../../../include/datasets/text_classification/sogou_news.h"

namespace xt::data::datasets {

    SogouNews::SogouNews(const std::string &root): SogouNews::SogouNews(root, DataMode::TRAIN, false) {
    }

    SogouNews::SogouNews(const std::string &root, DataMode mode): SogouNews::SogouNews(root, mode, false) {
    }

    SogouNews::SogouNews(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SogouNews: SogouNews not implemented");
    }


    SogouNews::SogouNews(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SogouNews: SogouNews not implemented");
    }


}
