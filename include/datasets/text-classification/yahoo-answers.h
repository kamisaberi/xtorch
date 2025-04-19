#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class YahooAnswers : BaseDataset {
        public :
            explicit YahooAnswers(const std::string &root);
        YahooAnswers(const std::string &root, DataMode mode);
        YahooAnswers(const std::string &root, DataMode mode , bool download);
        YahooAnswers(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
