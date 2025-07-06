#pragma once


#include "../common.h"


namespace xt::datasets {



    class TimeSeriesDataset : public xt::datasets::Dataset {
    public :
        TimeSeriesDataset(const std::string &file_path);
        TimeSeriesDataset(const std::string &file_path,DataMode mode);
        TimeSeriesDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
