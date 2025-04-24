#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class SpeechCommands : BaseDataset {
    public :
        explicit SpeechCommands(const std::string &root);

        SpeechCommands(const std::string &root, DataMode mode);

        SpeechCommands(const std::string &root, DataMode mode, bool download);

        SpeechCommands(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
