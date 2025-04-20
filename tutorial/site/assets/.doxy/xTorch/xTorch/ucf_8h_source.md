

# File ucf.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**video**](dir_424049e583f42f721b040286e87ec464.md) **>** [**ucf.h**](ucf_8h.md)

[Go to the documentation of this file](ucf_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
   class UCF101 : public BaseDataset {
       /*
            """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``. The dataset itself can be downloaded from the dataset website;
    annotations that ``annotation_path`` should be pointing to can be downloaded from `here
    <https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip>`_.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (str or ``pathlib.Path``): Root directory of the UCF101 Dataset.
        annotation_path (str): path to the folder containing the split files;
            see docstring above for download instructions of these files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
            -  audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
               and `L` is the number of points
            - label (int): class of the video clip
    """


        */
   public :
       explicit UCF101(const std::string &root);
       UCF101(const std::string &root, DataMode mode);
       UCF101(const std::string &root, DataMode mode , bool download);
       UCF101(const std::string &root, DataMode mode , bool download, TransformType transforms);

   private :
       void load_data();

       void check_resources();


    };
}
```


