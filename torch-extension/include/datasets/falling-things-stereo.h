#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class FallingThingsStereo : public BaseDataset {
    public :
        FallingThingsStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FallingThingsStereo(const fs::path &root, DatasetArguments args);

    private :
        fs::path  url ="https://drive.google.com/open?id=1y4h9T6D9rf6dAmsRwEtfzJdcghCnI_01";
        fs::path dataset_file_name = "fat.zip";
        std::string dataset_file_md5 = "????????????????????????????????";
        fs::path dataset_folder_name = "fat";
        void load_data();
        void check_resources();
    };
}
