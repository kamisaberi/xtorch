#pragma once

#include "../common.h"



// Forward-declare the cv::CLAHE class to avoid including heavy OpenCV headers here.
// This is a PIMPL (Pointer to Implementation) style approach.
namespace cv {
    class CLAHE;
}

namespace xt::transforms::image {

    /**
     * @class CLAHE
     * @brief An image transformation that applies Contrast Limited Adaptive
     *        Histogram Equalization.
     *
     * This is a powerful technique for enhancing local contrast in images. It works
     * especially well on images with regions that are both very dark and very bright.
     * This implementation uses OpenCV's CLAHE algorithm.
     *
     * Note: This transform operates on the L channel of the Lab color space for
     * color images, or directly on the single channel for grayscale images.
     */
    class CLAHE : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default CLAHE parameters.
         */
        CLAHE();

        /**
         * @brief Constructs the CLAHE transform with specific parameters.
         * @param clip_limit The threshold for contrast limiting.
         * @param tile_grid_size A vector of two ints {height, width} for the grid
         *                       size for histogram equalization.
         */
        CLAHE(double clip_limit, std::vector<int> tile_grid_size);

        /**
         * @brief Destructor to properly deallocate the CLAHE object.
         */
        ~CLAHE();

        /**
         * @brief Executes the CLAHE operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting contrast-enhanced torch::Tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        // --- PIMPL Support ---
        // Rule of 5 for proper resource management of the cv::CLAHE pointer
        CLAHE(const CLAHE& other); // Copy constructor
        CLAHE(CLAHE&& other) noexcept; // Move constructor
        CLAHE& operator=(const CLAHE& other); // Copy assignment
        CLAHE& operator=(CLAHE&& other) noexcept; // Move assignment

    private:
        // Using a pointer to the cv::CLAHE object to hide OpenCV implementation
        // details from the header file. This is the PIMPL idiom.
        cv::Ptr<cv::CLAHE> clahe_ptr_;
    };

} // namespace xt::transforms::image