#include "../../include/datasets/sintel.h"

namespace xt::data::datasets {

    // ---------------------- Sintel ---------------------- //

    Sintel::Sintel(const std::string &root): Sintel::Sintel(root, DataMode::TRAIN, false) {
    }

    Sintel::Sintel(const std::string &root, DataMode mode): Sintel::Sintel(root, mode, false) {
    }

    Sintel::Sintel(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Sintel: Sintel not implemented");
    }


    Sintel::Sintel(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Sintel: Sintel not implemented");
    }


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
