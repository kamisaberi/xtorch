#include "../../../include/datasets/natural-language-inference/qqp.h"

namespace xt::data::datasets {

    QQP::QQP(const std::string &root): QQP::QQP(root, DataMode::TRAIN, false) {
    }

    QQP::QQP(const std::string &root, DataMode mode): QQP::QQP(root, mode, false) {
    }

    QQP::QQP(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("QQP: QQP not implemented");
    }


    QQP::QQP(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("QQP: QQP not implemented");
    }


}
