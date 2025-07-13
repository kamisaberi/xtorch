#pragma once

#include "../common.h"


#include <torch/torch.h> // Assuming LibTorch is used for tensors
#include <memory>        // For std::shared_ptr

namespace xt::transforms::video {

    /**
     * @class FrameInterpolatorClient
     * @brief An abstract interface for a frame interpolation model/service.
     *
     * This contract allows the FrameInterpolation transform to be independent of
     * the specific deep learning model or inference engine being used.
     */
    class FrameInterpolatorClient {
    public:
        virtual ~FrameInterpolatorClient() = default;

        /**
         * @brief Generates intermediate frames between two input frames.
         * @param start_frame The first frame tensor of shape (C, H, W).
         * @param end_frame The second frame tensor of shape (C, H, W).
         * @param num_to_generate The number of intermediate frames to create.
         * @return A tensor of shape (num_to_generate, C, H, W) containing the
         *         newly created frames in temporal order.
         */
        virtual auto interpolate(
            const torch::Tensor& start_frame,
            const torch::Tensor& end_frame,
            int num_to_generate
        ) const -> torch::Tensor = 0;
    };


    /**
     * @class FrameInterpolation
     * @brief A video transformation that inserts a specified number of AI-generated
     *        frames between two existing frames to increase the frame rate.
     */
    class FrameInterpolation : public xt::Module {
    public:
        /**
         * @brief Constructs the FrameInterpolation transform.
         *
         * @param client A shared pointer to a concrete FrameInterpolatorClient.
         * @param num_to_insert The number of new frames to insert between each
         *                      pair of input frames. For example, a value of 1
         *                      will double the frame rate. Defaults to 1.
         */
        explicit FrameInterpolation(
            std::shared_ptr<FrameInterpolatorClient> client,
            int num_to_insert = 1
        );

        /**
         * @brief Executes the frame interpolation operation.
         * @param tensors An initializer list expected to contain exactly two
         *                torch::Tensor objects: the start frame and the end frame.
         *                Both tensors should have shape (C, H, W).
         * @return An std::any containing a torch::Tensor of shape
         *         (num_to_insert, C, H, W) holding the generated frames.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::shared_ptr<FrameInterpolatorClient> client_;
        int num_to_insert_;
    };

} // namespace xt::transforms::video