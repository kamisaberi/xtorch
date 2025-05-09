#pragma once

#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class VOCSegmentation : BaseDataset {
        /*
        """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

     Args:
         root (str or ``pathlib.Path``): Root directory of the VOC Dataset.
         year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
         image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
             ``year=="2007"``, can also be ``"test"``.
         download (bool, optional): If true, downloads the dataset from the internet and
             puts it in root directory. If dataset is already downloaded, it is not
             downloaded again.
         transform (callable, optional): A function/transform that  takes in a PIL image
             and returns a transformed version. E.g, ``transforms.RandomCrop``
         target_transform (callable, optional): A function/transform that takes in the
             target and transforms it.
         transforms (callable, optional): A function/transform that takes input sample and its target as entry
             and returns a transformed version.
     """

         */

    public :
        explicit VOCSegmentation(const std::string &root);
        VOCSegmentation(const std::string &root, DataMode mode);
        VOCSegmentation(const std::string &root, DataMode mode , bool download);
        VOCSegmentation(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
