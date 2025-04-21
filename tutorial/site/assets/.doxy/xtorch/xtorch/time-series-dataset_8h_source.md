

# File time-series-dataset.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**time-series-dataset.h**](time-series-dataset_8h.md)

[Go to the documentation of this file](time-series-dataset_8h.md)


```C++
#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class TimeSeriesDataset : public BaseDataset {
    public :
        TimeSeriesDataset(const std::string &file_path);
        TimeSeriesDataset(const std::string &file_path,DataMode mode);
        TimeSeriesDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedTimeSeriesDataset : public BaseDataset {
    public :
        StackedTimeSeriesDataset(const std::string &folder_path);
        StackedTimeSeriesDataset(const std::string &folder_path,DataMode mode);
        StackedTimeSeriesDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedTimeSeriesDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
```


