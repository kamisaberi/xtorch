#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class KittiFlow : BaseDataset {
        /*
        """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow>`__ dataset for optical flow (2015).

    The dataset is expected to have the following structure: ::

        root
            KittiFlow
                testing
                    image_2
                training
                    image_2
                    flow_occ

    Args:
        root (str or ``pathlib.Path``): Root directory of the KittiFlow Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
    """

         */
    public :
        explicit  KittiFlow(const std::string &root);
        KittiFlow(const std::string &root, DataMode mode);
        KittiFlow(const std::string &root, DataMode mode , bool download);
        KittiFlow(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };

}
