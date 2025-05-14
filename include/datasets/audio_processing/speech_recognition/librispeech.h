#pragma once

#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class LibriSpeech : BaseDataset {
    public :
        explicit LibriSpeech(const std::string &root);

        LibriSpeech(const std::string &root, DataMode mode);

        LibriSpeech(const std::string &root, DataMode mode, bool download);

        LibriSpeech(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
