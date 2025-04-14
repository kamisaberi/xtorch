#include "../../../include/datasets/image-classification/euro-sat.h"

namespace xt::data::datasets {

    EuroSAT::EuroSAT(const std::string &root): EuroSAT::EuroSAT(root, DataMode::TRAIN, false) {
    }

    EuroSAT::EuroSAT(const std::string &root, DataMode mode): EuroSAT::EuroSAT(root, mode, false) {
    }

    EuroSAT::EuroSAT(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("EuroSAT: EuroSAT not implemented");
    }


    EuroSAT::EuroSAT(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("EuroSAT: EuroSAT not implemented");
    }

}
