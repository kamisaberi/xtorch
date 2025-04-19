#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CommonVoice : BaseDataset {
    public :
        explicit CommonVoice(const std::string &root);
        CommonVoice(const std::string &root, DataMode mode);
        CommonVoice(const std::string &root, DataMode mode , bool download);
        CommonVoice(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
