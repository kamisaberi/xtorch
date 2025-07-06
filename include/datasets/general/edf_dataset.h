#pragma once


#include "../common.h"



namespace xt::datasets {

    class EDFDataset : public xt::datasets::Dataset {
    public :
        EDFDataset(const std::string &file_path);
        EDFDataset(const std::string &file_path,DataMode mode);
        EDFDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> target_transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
