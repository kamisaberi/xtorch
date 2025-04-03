#include "../../../include/datasets/specific/cre-stereo.h"

namespace xt::data::datasets {
    // ---------------------- CarlaStereo ---------------------- //
    CarlaStereo::CarlaStereo(const std::string &root): CarlaStereo::CarlaStereo(root, DataMode::TRAIN, false) {
    }

    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode): CarlaStereo::CarlaStereo(root, mode, false) {
    }

    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CarlaStereo: CarlaStereo not implemented");
    }


    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CarlaStereo: CarlaStereo not implemented");
    }


    // ---------------------- CREStereo ---------------------- //

    CREStereo::CREStereo(const std::string &root): CREStereo::CREStereo(root, DataMode::TRAIN, false) {
    }

    CREStereo::CREStereo(const std::string &root, DataMode mode): CREStereo::CREStereo(root, mode, false) {
    }

    CREStereo::CREStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CREStereo: CREStereo not implemented");
    }


    CREStereo::CREStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CREStereo: CREStereo not implemented");
    }
}
