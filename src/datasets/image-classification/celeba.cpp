#include "../../../include/datasets/image-classification/celeba.h"

namespace xt::data::datasets {

    CelebA::CelebA(const std::string &root): CelebA::CelebA(root, DataMode::TRAIN, false) {
    }

    CelebA::CelebA(const std::string &root, DataMode mode): CelebA::CelebA(root, mode, false) {
    }

    CelebA::CelebA(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CelebA: CelebA not implemented");
    }


    CelebA::CelebA(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CelebA: CelebA not implemented");
    }

}
