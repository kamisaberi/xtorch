#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class InStereo2k : BaseDataset {
        /*
        """`InStereo2k <https://github.com/YuhuaXu/StereoDataset>`_ dataset.

        The dataset is expected to have the following structure: ::

            root
                InStereo2k
                    train
                        scene1
                            left.png
                            right.png
                            left_disp.png
                            right_disp.png
                            ...
                        scene2
                        ...
                    test
                        scene1
                            left.png
                            right.png
                            left_disp.png
                            right_disp.png
                            ...
                        scene2
                        ...

        Args:
            root (str or ``pathlib.Path``): Root directory where InStereo2k is located.
            split (string): Either "train" or "test".
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """
         */
    public :
        InStereo2k(const std::string &root);
        InStereo2k(const std::string &root, DataMode mode);
        InStereo2k(const std::string &root, DataMode mode , bool download);
        InStereo2k(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();
        void check_resources();
    };
}
