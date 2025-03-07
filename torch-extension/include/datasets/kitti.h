#pragma once

#include "../base/datasets.h"


namespace torch::ext::data::datasets {
    class Kitti : torch::data::Dataset<Kitti> {

    private:
        fs::path data_url = fs::path("https://s3.eu-central-1.amazonaws.com/avg-kitti/");
        vector<std::string> resources = {
                "data_object_image_2.zip",
                "data_object_label_2.zip"
        };
        fs::path image_dir_name = fs::path("image_2");
        fs::path labels_dir_name = fs::path("label_2");

    public :
        Kitti();
    };

    class KittiFlow : torch::data::Dataset<KittiFlow> {

    public :
        KittiFlow();
    };

    class Kitti2012Stereo : torch::data::Dataset<Kitti2012Stereo> {

    public :
        Kitti2012Stereo();
    };

    class Kitti2015Stereo : torch::data::Dataset<Kitti2015Stereo> {

    public :
        Kitti2015Stereo();
    };
}
