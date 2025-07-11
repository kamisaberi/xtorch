#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class MotionBlur
     * @brief An image transformation that applies a motion blur effect.
     *
     * This filter simulates the blur caused by camera or object motion by
     * convolving the image with a kernel representing a line segment.
     * This implementation uses OpenCV for the underlying computation.
     */
    class MotionBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a moderate kernel size of 7.
         */
        MotionBlur();

        /**
         * @brief Constructs the MotionBlur transform.
         * @param kernel_size The size of the motion blur kernel, which corresponds to
         *                    the length of the motion trail. Must be an odd integer.
         * @param angle The angle of the motion in degrees. -1 means a random angle
         *              will be chosen for each application.
         * @param direction The direction of motion, either forward (0), backward (-1),
         *                  or both (1). Both is the most common.
         */
        MotionBlur(int kernel_size, double angle = -1.0, int direction = 1);

        /**
         * @brief Executes the motion blur operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int kernel_size_;
        double angle_;
        int direction_;
    };

} // namespace xt::transforms::image