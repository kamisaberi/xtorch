#include "../../include/datasets/oxfordIII-t-pet.h"

namespace xt::data::datasets {

    OxfordIIITPet::OxfordIIITPet(const std::string &root): OxfordIIITPet::OxfordIIITPet(root, DataMode::TRAIN, false) {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string &root, DataMode mode): OxfordIIITPet::OxfordIIITPet(root, mode, false) {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("OxfordIIITPet: OxfordIIITPet not implemented");
    }


    OxfordIIITPet::OxfordIIITPet(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("OxfordIIITPet: OxfordIIITPet not implemented");
    }

}
