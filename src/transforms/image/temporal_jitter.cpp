#include "include/transforms/image/temporal_jitter.h"
//
// #include "transforms/video/temporal_jitter.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy video tensor [T, C, H, W]
//     // Let's make each frame a different solid color to easily track the shift.
//     int num_frames = 16;
//     torch::Tensor video = torch::zeros({num_frames, 3, 32, 32});
//     for (int t = 0; t < num_frames; ++t) {
//         // Let's make frame 't' have a value of 't'.
//         video[t] = static_cast<float>(t + 1);
//     }
//
//     std::cout << "Original video's first frame mean: " << video[0].mean().item<float>() << std::endl;
//     std::cout << "Original video's last frame mean: " << video[num_frames - 1].mean().item<float>() << std::endl;
//
//     // 2. Instantiate the transform with a max jitter of 3 frames
//     xt::transforms::video::TemporalJitter jitterer(3);
//
//     // 3. Apply the transform
//     std::any result_any = jitterer.forward({video});
//     torch::Tensor jittered_video = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "\n--- After Jitter ---" << std::endl;
//     std::cout << "Jittered video shape is unchanged: " << jittered_video.sizes() << std::endl;
//     std::cout << "Jittered video's first frame mean: " << jittered_video[0].mean().item<float>() << std::endl;
//     std::cout << "Jittered video's last frame mean: " << jittered_video[num_frames - 1].mean().item<float>() << std::endl;
//
//     // Depending on the random offset, the first or last frame's mean value will have changed.
//     // For example, if the offset was +2:
//     // The new first frame is the original frame 2 (value=3), and the last two frames
//     // are duplicates of the original last frame (value=16).
//     // So the new first frame mean would be 3.0 and the new last frame mean would be 16.0.
//     //
//     // If the offset was -2:
//     // The first two frames are duplicates of the original first frame (value=1).
//     // The new last frame is the original frame 13 (value=14).
//     // So the new first frame mean would be 1.0 and the new last frame mean would be 14.0.
//
//     return 0;
// }
//
//

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