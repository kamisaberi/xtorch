#pragma once

#include "datasets/base/base.h"

namespace xt::data::datasets {
    class WikiTextV2 : BaseDataset {
        public :
            explicit WikiTextV2(const std::string &root);
        WikiTextV2(const std::string &root, DataMode mode);
        WikiTextV2(const std::string &root, DataMode mode , bool download);
        WikiTextV2(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
