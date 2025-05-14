#include "datasets/computer_vision/stereo_matching/sintel_stereo.h"

namespace xt::data::datasets {


    // ---------------------- SintelStereo ---------------------- //

    SintelStereo::SintelStereo(const std::string &root): SintelStereo::SintelStereo(root, DataMode::TRAIN, false) {
    }

    SintelStereo::SintelStereo(const std::string &root, DataMode mode): SintelStereo::SintelStereo(root, mode, false) {
    }

    SintelStereo::SintelStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SintelStereo: SintelStereo not implemented");
    }


    SintelStereo::SintelStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SintelStereo: SintelStereo not implemented");
    }

}
