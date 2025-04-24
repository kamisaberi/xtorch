#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


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
