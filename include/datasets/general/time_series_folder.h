#pragma once


#include "../common.h"


namespace xt::datasets
{
    class TimeSeriesFolder : public xt::datasets::Dataset
    {
    public :
        TimeSeriesFolder(const std::string& file_path);
        TimeSeriesFolder(const std::string& file_path, DataMode mode);
        TimeSeriesFolder(const std::string& file_path, DataMode mode, std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };
}
