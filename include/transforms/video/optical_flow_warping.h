#pragma once

#include "../common.h"

#include <opencv2/opencv.hpp> // Assumes OpenCV is installed
#include <memory>             // For std::shared_ptr

namespace xt::transforms::video {

    /**
     * @class OpticalFlowClient
     * @brief An abstract interface for an optical flow calculation algorithm.
     *
     * This contract allows the main transform to be independent of the specific
     * flow algorithm used (e.g., Farnebäck, RAFT, PWC-Net).
     */
    class OpticalFlowClient {
    public:
        virtual ~OpticalFlowClient() = default;

        /**
         * @brief Calculates the dense optical flow between two frames.
         * @param frame1 The first frame (grayscale, 8-bit).
         * @param frame2 The second frame (grayscale, 8-bit).
         * @return A cv::Mat of type CV_32FC2 representing the flow field, where
         *         each element is a 2D vector (dx, dy) of the pixel's motion.
         */
        virtual auto calculate_flow(const cv::Mat& frame1, const cv::Mat& frame2) const -> cv::Mat = 0;
    };


    /**
     * @class OpticalFlowWarping
     * @brief A video transformation that warps a frame using an optical flow field.
     *
     * This transform calculates the optical flow from a source frame to a target
     * frame and then applies this flow to warp the source frame, attempting to
     * make it look like the target.
     */
    class OpticalFlowWarping : public xt::Module {
    public:
        /**
         * @brief Constructs the OpticalFlowWarping transform.
         * @param client A shared pointer to a concrete OpticalFlowClient.
         */
        explicit OpticalFlowWarping(std::shared_ptr<OpticalFlowClient> client);

        /**
         * @brief Executes the optical flow warping operation.
         * @param tensors An initializer list expected to contain exactly two
         *                torch::Tensor objects: the source frame to be warped,
         *                and the target frame to align to.
         * @return An std::any containing the warped source frame as a torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::shared_ptr<OpticalFlowClient> client_;
    };

} // namespace xt::transforms::video