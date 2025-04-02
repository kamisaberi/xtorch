#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class FER2013 : public BaseDataset {
        /*
            `FER2013
            <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

            .. note::
                This dataset can return test labels only if ``fer2013.csv`` OR
                ``icml_face_data.csv`` are present in ``root/fer2013/``. If only
                ``train.csv`` and ``test.csv`` are present, the test labels are set to
                ``None``.

            Args:
                root (str or ``pathlib.Path``): Root directory of dataset where directory
                    ``root/fer2013`` exists. This directory may contain either
                    ``fer2013.csv``, ``icml_face_data.csv``, or both ``train.csv`` and
                    ``test.csv``. Precendence is given in that order, i.e. if
                    ``fer2013.csv`` is present then the rest of the files will be
                    ignored. All these (combinations of) files contain the same data and
                    are supported for convenience, but only ``fer2013.csv`` and
                    ``icml_face_data.csv`` are able to return non-None test labels.
                split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
                transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
                    version. E.g, ``transforms.RandomCrop``
                target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        */
    public :
        FER2013(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FER2013(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
