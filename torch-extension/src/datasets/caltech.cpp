#include "../../include/datasets/caltech.h"

namespace xt::data::datasets {
    Caltech101::Caltech101(const std::string &root): Caltech101::Caltech101(root, DataMode::TRAIN, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode): Caltech101::Caltech101(root, mode, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
    }


    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download,
                           vector<std::function<torch::Tensor(torch::Tensor)> > transforms) : BaseDataset(
        root, mode, download, transforms) {
    }

    // Caltech101::Caltech101() {
    //     throw NotImplementedException();
    // }
    // Caltech256::Caltech256() {
    //     throw NotImplementedException();
    // }
}
