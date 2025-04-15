#include "../../../include/datasets/specific/eth-3d-stereo.h"

namespace xt::data::datasets {

    ETH3DStereo::ETH3DStereo(const std::string &root): ETH3DStereo::ETH3DStereo(root, DataMode::TRAIN, false) {
    }

    ETH3DStereo::ETH3DStereo(const std::string &root, DataMode mode): ETH3DStereo::ETH3DStereo(root, mode, false) {
    }

    ETH3DStereo::ETH3DStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("ETH3DStereo: ETH3DStereo not implemented");
    }


    ETH3DStereo::ETH3DStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("ETH3DStereo: ETH3DStereo not implemented");
    }

}
