# Video Transforms

Video data adds a temporal dimension to the challenges of computer vision. A video is a sequence of image frames, and processing this data requires handling both the spatial content of each frame and the temporal relationships between them.

Video transforms are designed to operate on these sequences of frames. They are essential for preparing video clips for input to models like video classifiers or action recognition networks.

All video transforms are located under the `xt::transforms` namespace and can be found in the `<xtorch/transforms/video/>` header directory.

## General Usage

Video transforms are used in a `Compose` pipeline, just like image transforms. However, they expect their input to be a tensor with a temporal dimension, typically with a shape like `[Time, Channels, Height, Width]`.

A common workflow involves:
1.  Loading a video and decoding it into a sequence of frames.
2.  Applying temporal transforms to select or sample frames.
3.  Applying spatial (image) transforms to each of the selected frames.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Define a pipeline of video and image transformations
    auto video_pipeline = std::make_unique<xt::transforms::Compose>(
        // --- Temporal Transforms ---
        // Sample 16 frames uniformly from the video clip
        std::make_shared<xt::transforms::video::UniformTemporalSubsample>(16),
        // Randomly reverse the sequence of the 16 frames with a 50% probability
        std::make_shared<xt::transforms::video::RandomClipReverse>(0.5),

        // --- Spatial Transforms (applied to each frame) ---
        // Note: You would typically wrap spatial transforms to apply them per-frame.
        // For simplicity, we assume the dataset handles this application logic.
        std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{128, 128}),
        std::make_shared<xt::transforms::general::Normalize>(
            std::vector<float>{0.5, 0.5, 0.5},
            std::vector<float>{0.5, 0.5, 0.5}
        )
    );

    // 2. Pass the pipeline to a video Dataset
    // The dataset will apply these transforms to each video it loads.
    // auto dataset = xt::datasets::UCF101("./data", std::move(video_pipeline));

    // 3. The DataLoader will yield batches of processed video clips
    // xt::dataloaders::ExtendedDataLoader data_loader(dataset, 8);
    // for (auto& batch : data_loader) {
    //     auto video_clips = batch.first; // Shape e.g.,
    // }
}
```

---

## Available Video Transforms

xTorch provides the following transforms for video data:

### Temporal Transforms

These transforms operate along the time dimension of a video clip.

| Transform | Description | Header File |
|---|---|---|
| `UniformTemporalSubsample`| Subsamples a fixed number of frames from a video clip at a uniform frame rate. This is a common way to create a fixed-size input from videos of varying lengths. | `uniform_temporal_subsample.h` |
| `RandomClipReverse` | Randomly reverses the order of frames in a clip with a given probability. A simple form of temporal data augmentation. | `random_clip_reverse.h` |

### Spatio-Temporal Transforms

These transforms modify both the spatial and temporal aspects of the video.

| Transform | Description | Header File |
|---|---|---|
| `FrameInterpolation`| Generates intermediate frames between existing ones, which can be used to increase the frame rate or for slow-motion effects. | `frame_interpolation.h` |
| `OpticalFlowWarping`| Warps video frames based on calculated optical flow, a technique used in video compression and frame rate conversion. | `optical_flow_warping.h` |

!!! info "Applying Image Transforms to Videos"
To apply a standard image transform (like `RandomCrop` or `ColorJitter`) to every frame of a video, you typically need to iterate over the time dimension and apply the transform to each `[C, H, W]` frame individually. The `Compose` applier can be used to chain these operations. Support for per-frame application wrappers may be included in the future.
