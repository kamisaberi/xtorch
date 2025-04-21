

# File kinetics.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**video**](dir_424049e583f42f721b040286e87ec464.md) **>** [**kinetics.h**](kinetics_8h.md)

[Go to the documentation of this file](kinetics_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Kinetics : BaseDataset {
        /*
        https://s3.amazonaws.com/kinetics
            """`Generic Kinetics <https://www.deepmind.com/open-source/kinetics>`_
    dataset.

    Kinetics-400/600/700 are action recognition video datasets.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Kinetics Dataset.
            Directory should be structured as follows:
            .. code::

                root/
                ├── split
                │   ├──  class1
                │   │   ├──  vid1.mp4
                │   │   ├──  vid2.mp4
                │   │   ├──  vid3.mp4
                │   │   ├──  ...
                │   ├──  class2
                │   │   ├──   vidx.mp4
                │   │    └── ...

            Note: split is appended automatically using the split argument.
        frames_per_clip (int): number of frames in a clip
        num_classes (int): select between Kinetics-400 (default), Kinetics-600, and Kinetics-700
        split (str): split of the dataset to consider; supports ``"train"`` (default) ``"val"`` ``"test"``
        frame_rate (float): If omitted, interpolate different frame rate for each clip.
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
        download (bool): Download the official version of the dataset to root folder.
        num_workers (int): Use multiple workers for VideoClips creation
        num_download_workers (int): Use multiprocessing in order to speed up download.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" or "TCHW" (default).
            Note that in most other utils and datasets, the default is actually "THWC".

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, C, H, W] or Tensor[T, H, W, C]): the `T` video frames in torch.uint8 tensor
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points in torch.float tensor
            - label (int): class of the video clip

    Raises:
        RuntimeError: If ``download is True`` and the video archives are already extracted.
    """


         */
    public :
        explicit Kinetics(const std::string &root);
        Kinetics(const std::string &root, DataMode mode);
        Kinetics(const std::string &root, DataMode mode , bool download);
        Kinetics(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
    //     _TAR_URLS = {
    //     "400": "https://s3.amazonaws.com/kinetics/400/{split}/k400_{split}_path.txt",
    //     "600": "https://s3.amazonaws.com/kinetics/600/{split}/k600_{split}_path.txt",
    //     "700": "https://s3.amazonaws.com/kinetics/700_2020/{split}/k700_2020_{split}_path.txt",
    // }
    //     _ANNOTATION_URLS = {
    //     "400": "https://s3.amazonaws.com/kinetics/400/annotations/{split}.csv",
    //     "600": "https://s3.amazonaws.com/kinetics/600/annotations/{split}.csv",
    //     "700": "https://s3.amazonaws.com/kinetics/700_2020/annotations/{split}.csv",
    // }

        void load_data();

        void check_resources();
    };
}
```


