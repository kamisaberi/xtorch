#pragma once


#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class CarlaStereo : public  BaseDataset {
    public :
        CarlaStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        CarlaStereo(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };


    class CREStereo : public BaseDataset {
    public :
        CREStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        CREStereo(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
