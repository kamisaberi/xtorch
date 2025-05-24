#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class Kitti2012Stereo : xt::datasets::Dataset
    {
        /*
        """
    KITTI dataset from the `2012 stereo evaluation benchmark <http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php>`_.
    Uses the RGB images for consistency with KITTI 2015.

    The dataset is expected to have the following structure: ::

        root
            Kitti2012
                testing
                    colored_0
                        1_10.png
                        2_10.png
                        ...
                    colored_1
                        1_10.png
                        2_10.png
                        ...
                training
                    colored_0
                        1_10.png
                        2_10.png
                        ...
                    colored_1
                        1_10.png
                        2_10.png
                        ...
                    disp_noc
                        1.png
                        2.png
                        ...
                    calib

    Args:
        root (str or ``pathlib.Path``): Root directory where `Kitti2012` is located.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

         */
    public :
        explicit Kitti2012Stereo(const std::string& root);
        Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode);
        Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download);
        Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                        std::unique_ptr<xt::Module> transformer);
        Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                        std::unique_ptr<xt::Module> transformer,
                        std::unique_ptr<xt::Module> target_transformer);

    private :
        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
