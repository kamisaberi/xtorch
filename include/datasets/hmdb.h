#pragma once

#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class HMDB51 : BaseDataset {
        /*
        """
            `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
            dataset.

            HMDB51 is an action recognition video dataset.
            This dataset consider every video as a collection of video clips of fixed size, specified
            by ``frames_per_clip``, where the step in frames between each clip is given by
            ``step_between_clips``.

            To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
            and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
            elements will come from video 1, and the next three elements from video 2.
            Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
            frames in a video might be present.

            Internally, it uses a VideoClips object to handle clip creation.

            Args:
                root (str or ``pathlib.Path``): Root directory of the HMDB51 Dataset.
                annotation_path (str): Path to the folder containing the split files.
                frames_per_clip (int): Number of frames in a clip.
                step_between_clips (int): Number of frames between each clip.
                fold (int, optional): Which fold to use. Should be between 1 and 3.
                train (bool, optional): If ``True``, creates a dataset from the train split,
                    otherwise from the ``test`` split.
                transform (callable, optional): A function/transform that takes in a TxHxWxC video
                    and returns a transformed version.
                output_format (str, optional): The format of the output video tensors (before transforms).
                    Can be either "THWC" (default) or "TCHW".

            Returns:
                tuple: A 3-tuple with the following entries:

                    - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
                    - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
                      and `L` is the number of points
                    - label (int): class of the video clip
            """
         */
    public :
        HMDB51(const std::string &root);
        HMDB51(const std::string &root, DataMode mode);
        HMDB51(const std::string &root, DataMode mode , bool download);
        HMDB51(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        // data_url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
        // splits = {
        //         "url": "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        //         "md5": "15e67781e70dcfbdce2d7dbb9b3344b5",
        // }

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
