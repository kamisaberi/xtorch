#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class EDFDataset : public BaseDataset {
    public :
        EDFDataset(const std::string &file_path);
        EDFDataset(const std::string &file_path,DataMode mode);
        EDFDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedEDFDataset : public BaseDataset {
    public :
        StackedEDFDataset(const std::string &folder_path);
        StackedEDFDataset(const std::string &folder_path,DataMode mode);
        StackedEDFDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedEDFDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
