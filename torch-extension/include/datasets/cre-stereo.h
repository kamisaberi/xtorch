#pragma once


#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class CarlaStereo : public  BaseDataset {
    public :
        CarlaStereo(const std::string &root);
        CarlaStereo(const std::string &root, DataMode mode);
        CarlaStereo(const std::string &root, DataMode mode , bool download);
        CarlaStereo(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data();

        void check_resources();
    };


    class CREStereo : public BaseDataset {
    public :
        CREStereo(const std::string &root);
        CREStereo(const std::string &root, DataMode mode);
        CREStereo(const std::string &root, DataMode mode , bool download);
        CREStereo(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data();

        void check_resources();
    };
}
