#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class DBPedia : BaseDataset {
        public :
            explicit DBPedia(const std::string &root);
        DBPedia(const std::string &root, DataMode mode);
        DBPedia(const std::string &root, DataMode mode , bool download);
        DBPedia(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
