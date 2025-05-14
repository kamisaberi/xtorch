#include "datasets/computer_vision/stereo_matching/cre_stereo.h"

namespace xt::data::datasets {


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
