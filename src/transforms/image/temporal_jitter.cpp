#include "include/transforms/image/temporal_jitter.h"


namespace xt::transforms::video {

    TemporalJitter::TemporalJitter() : max_jitter_frames_(4) {}

    TemporalJitter::TemporalJitter(int max_jitter_frames) : max_jitter_frames_(max_jitter_frames) {
        if (max_jitter_frames_ < 0) {
            throw std::invalid_argument("max_jitter_frames must be non-negative.");
        }
    }

    auto TemporalJitter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TemporalJitter::forward received an empty list of tensors.");
        }
        torch::Tensor video_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!video_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to TemporalJitter is not defined.");
        }
        if (video_tensor.dim() != 4) {
            throw std::invalid_argument("TemporalJitter expects a 4D video tensor of shape [T, C, H, W].");
        }

        // If jitter is 0, do nothing.
        if (max_jitter_frames_ == 0) {
            return video_tensor;
        }

        const int64_t T = video_tensor.size(0); // Number of frames

        // 2. --- Calculate Random Temporal Shift ---
        // Generate a random offset between -max_jitter_frames_ and +max_jitter_frames_
        int64_t offset = torch::randint(-max_jitter_frames_, max_jitter_frames_ + 1, {1}).item<int64_t>();

        // 3. --- Apply the Shift using Slicing and Padding ---
        torch::Tensor jittered_video;

        if (offset == 0) {
            // No shift needed
            return video_tensor;
        } else if (offset > 0) {
            // Shift forward in time: Drop `offset` frames from the beginning.
            // The new video starts at `offset` and has length `T - offset`.
            auto sliced_video = video_tensor.slice(/*dim=*/0, /*start=*/offset, /*end=*/T);

            // Pad the end with duplicates of the last frame to maintain original length.
            auto last_frame = sliced_video.slice(0, -1, T).repeat({offset, 1, 1, 1});
            jittered_video = torch::cat({sliced_video, last_frame}, 0);

        } else { // offset < 0
            // Shift backward in time: Drop `abs(offset)` frames from the end.
            int64_t abs_offset = -offset;
            // The new video has length `T - abs(offset)`.
            auto sliced_video = video_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/T - abs_offset);

            // Pad the beginning with duplicates of the first frame.
            auto first_frame = sliced_video.slice(0, 0, 1).repeat({abs_offset, 1, 1, 1});
            jittered_video = torch::cat({first_frame, sliced_video}, 0);
        }

        return jittered_video;
    }

} // namespace xt::transforms::video