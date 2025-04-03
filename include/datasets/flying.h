#pragma once

#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class FlyingChairs : public BaseDataset {
        /*
        """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow.
        You will also need to download the FlyingChairs_train_val.txt file from the dataset page.
        The dataset is expected to have the following structure: ::

            root
                FlyingChairs
                    data
                        00001_flow.flo
                        00001_img1.ppm
                        00001_img2.ppm
                        ...
                    FlyingChairs_train_val.txt
        Args:
            root (str or ``pathlib.Path``): Root directory of the FlyingChairs Dataset.
            split (string, optional): The dataset split, either "train" (default) or "val"
            transforms (callable, optional): A function/transform that takes in
                ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                ``valid_flow_mask`` is expected for consistency with other datasets which
                return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
        """
        */
    public :
        FlyingChairs(const std::string &root);
        FlyingChairs(const std::string &root, DataMode mode);
        FlyingChairs(const std::string &root, DataMode mode , bool download);
        FlyingChairs(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };

    class FlyingThings3D : public BaseDataset {
        /*
        """`FlyingThings3D <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>`_ dataset for optical flow.
       The dataset is expected to have the following structure: ::
            root
                FlyingThings3D
                    frames_cleanpass
                        TEST
                        TRAIN
                    frames_finalpass
                        TEST
                        TRAIN
                    optical_flow
                        TEST
                        TRAIN

        Args:
            root (str or ``pathlib.Path``): Root directory of the intel FlyingThings3D Dataset.
            split (string, optional): The dataset split, either "train" (default) or "test"
            pass_name (string, optional): The pass to use, either "clean" (default) or "final" or "both". See link above for
                details on the different passes.
            camera (string, optional): Which camera to return images from. Can be either "left" (default) or "right" or "both".
            transforms (callable, optional): A function/transform that takes in
                ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                ``valid_flow_mask`` is expected for consistency with other datasets which
                return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
        """
    */
    public :
        FlyingThings3D(const std::string &root);
        FlyingThings3D(const std::string &root, DataMode mode);
        FlyingThings3D(const std::string &root, DataMode mode , bool download);
        FlyingThings3D(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
