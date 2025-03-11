#pragma once

#include "base.h"
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
    class Kitti : BaseDataset {
    public :
        Kitti(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Kitti(const fs::path &root, DatasetArguments args);

    private:
        fs::path data_url = fs::path("https://s3.eu-central-1.amazonaws.com/avg-kitti/");
        vector<std::string> resources = {
            "data_object_image_2.zip",
            "data_object_label_2.zip"
        };
        fs::path image_dir_name = fs::path("image_2");
        fs::path labels_dir_name = fs::path("label_2");

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class KittiFlow : BaseDataset {
    public :
        KittiFlow(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        KittiFlow(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class Kitti2012Stereo : BaseDataset {
    public :
        Kitti2012Stereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Kitti2012Stereo(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class Kitti2015Stereo : torch::data::Dataset<Kitti2015Stereo> {
    public :
        Kitti2015Stereo();
    };
}
