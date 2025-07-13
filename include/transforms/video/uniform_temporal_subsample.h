#pragma once

#include "../common.h"
namespace xt::transforms::video {

    /**
     * @class UniformTemporalSubsample
     * @brief A video transformation that subsamples a fixed number of frames
     *        uniformly from a clip.
     *
     * Given a clip with T frames, this transform selects N frames from it such
     * that they are evenly spaced in time. This is a standard preprocessing
     * step for video models that expect a fixed-size temporal input.
     */
    class UniformTemporalSubsample : public xt::Module {
    public:
        /**
         * @brief Constructs the UniformTemporalSubsample transform.
         *
         * @param num_samples The number of frames to select from the clip.
         */
        explicit UniformTemporalSubsample(int num_samples);

        /**
         * @brief Executes the uniform subsampling operation.
         * @param tensors An initializer list expected to contain a single video
         *                clip as a torch::Tensor of shape (T, C, H, W), where T
         *                is the original number of frames.
         * @return An std::any containing the subsampled clip as a torch::Tensor
         *         with shape (num_samples, C, H, W).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_samples_;
    };

} // namespace xt::transforms::video