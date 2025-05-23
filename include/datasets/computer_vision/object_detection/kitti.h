#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class Kitti : public xt::datasets::Dataset{
        /*
        """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

         */
    public :
        explicit Kitti(const std::string& root);
        Kitti(const std::string& root, xt::datasets::DataMode mode);
        Kitti(const std::string& root, xt::datasets::DataMode mode, bool download);
        Kitti(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        Kitti(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private:
        fs::path url = fs::path("https://s3.eu-central-1.amazonaws.com/avg-kitti/");
        vector<std::string> resources = {
            "data_object_image_2.zip",
            "data_object_label_2.zip"
        };
        fs::path image_dir_name = fs::path("image_2");
        fs::path labels_dir_name = fs::path("label_2");
        fs::path dataset_folder_name = "kitti";


        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();

    };


}
