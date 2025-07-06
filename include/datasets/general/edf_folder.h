#pragma once


#include "../common.h"



namespace xt::datasets {

    class EDFFolder : public xt::datasets::Dataset {
    public :
        EDFFolder(const std::string &file_path);
        EDFFolder(const std::string &file_path,DataMode mode);
        EDFFolder(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> target_transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };





}
