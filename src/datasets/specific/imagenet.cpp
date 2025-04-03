#include "../../../include/datasets/specific/imagenet.h"

namespace xt::data::datasets {

    ImageNet::ImageNet(const std::string &root): ImageNet::ImageNet(root, DataMode::TRAIN, false) {
    }

    ImageNet::ImageNet(const std::string &root, DataMode mode): ImageNet::ImageNet(root, mode, false) {
    }

    ImageNet::ImageNet(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("ImageNet: ImageNet not implemented");
    }


    ImageNet::ImageNet(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("ImageNet: ImageNet not implemented");
    }


}
