#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class QQP : BaseDataset {
        public :
            explicit QQP(const std::string &root);
        QQP(const std::string &root, DataMode mode);
        QQP(const std::string &root, DataMode mode , bool download);
        QQP(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
