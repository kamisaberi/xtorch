#pragma once

#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class Kitti2015Stereo : BaseDataset {
        /*
        """
    KITTI dataset from the `2015 stereo evaluation benchmark <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php>`_.

    The dataset is expected to have the following structure: ::

        root
            Kitti2015
                testing
                    image_2
                        img1.png
                        img2.png
                        ...
                    image_3
                        img1.png
                        img2.png
                        ...
                training
                    image_2
                        img1.png
                        img2.png
                        ...
                    image_3
                        img1.png
                        img2.png
                        ...
                    disp_occ_0
                        img1.png
                        img2.png
                        ...
                    disp_occ_1
                        img1.png
                        img2.png
                        ...
                    calib

    Args:
        root (str or ``pathlib.Path``): Root directory where `Kitti2015` is located.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

         */
    public :
        explicit Kitti2015Stereo(const std::string &root);
        Kitti2015Stereo(const std::string &root, DataMode mode);
        Kitti2015Stereo(const std::string &root, DataMode mode , bool download);
        Kitti2015Stereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
