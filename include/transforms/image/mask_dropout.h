#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class MaskDropout
     * @brief An image transformation that randomly erases pixels within a provided mask.
     *
     * This augmentation technique takes an image and a corresponding binary mask.
     * It randomly selects a portion of the foreground pixels (where the mask is > 0)
     * and sets them to a specific fill value. This simulates occlusions on the
     * object of interest, forcing the model to learn from partial information.
     */
    class MaskDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Drops 10% of the masked area.
         */
        MaskDropout();

        /**
         * @brief Constructs the MaskDropout transform.
         * @param max_objects The maximum number of objects to apply dropout to.
         * @param p The probability of applying dropout to any given object.
         * @param holes_nb The number of holes to create in the mask.
         * @param mask_fill_value The value to fill the holes in the mask with.
         * @param p_replace The probability of replacing the object with the fill value.
         */
        MaskDropout(int max_objects, float p, int holes_nb, int mask_fill_value, float p_replace);

        /**
         * @brief Executes the mask dropout operation.
         * @param tensors An initializer list expected to contain two tensors:
         *                1. The image tensor (3D, [C, H, W])
         *                2. The mask tensor (2D [H, W] or 3D [1, H, W])
         * @return An std::any containing the resulting torch::Tensor with parts
         *         of the masked region erased. The original mask is not returned.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_objects_;
        float p_;
        int holes_nb_;
        int mask_fill_value_;
        float p_replace_;
    };

} // namespace xt::transforms::image