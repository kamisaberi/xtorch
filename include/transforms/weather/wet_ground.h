#pragma once

#include "../common.h"

#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class WetGround
     * @brief A transform that makes ground surfaces in an image appear wet.
     *
     * This transform applies darkening and specular reflections to areas defined by a
     * ground map to simulate the appearance of wet ground, such as after rainfall.
     * It can create effects ranging from damp surfaces to deep puddles reflecting
     * the sky.
     */
    class WetGround : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderately wet ground effect with gray puddle reflections.
         */
        WetGround();

        /**
         * @brief Constructs the WetGround transform with custom parameters.
         * @param darkening A factor from [0, 1] controlling how much the wet ground darkens.
         * @param specular_intensity A factor from [0, 1] controlling the shininess of damp surfaces.
         * @param puddle_threshold A value from [0, 1]. Areas in the ground map with values
         *                         above this threshold will become reflective puddles.
         * @param reflection_intensity A factor from [0, 1] controlling the opacity of puddle reflections.
         * @param reflection_color A 3-element tensor for the R, G, B color reflected in puddles (e.g., a sky color).
         */
        explicit WetGround(
                float darkening,
                float specular_intensity,
                float puddle_threshold,
                float reflection_intensity,
                torch::Tensor reflection_color
        );

        /**
         * @brief Executes the wet ground effect.
         * @param tensors An initializer list expected to contain two tensors:
         *                1. Image Tensor (C, H, W) - The source image.
         *                2. Ground Spec Map (1, H, W) or (H, W) - A map where non-zero values
         *                   indicate ground. The value's magnitude (0 to 1) should represent
         *                   puddle potential (e.g., inverted height, porosity).
         * @return An std::any containing the resulting torch::Tensor with the wet ground effect.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float darkening_;
        float specular_intensity_;
        float puddle_threshold_;
        float reflection_intensity_;
        torch::Tensor reflection_color_;
    };

} // namespace xt::transforms::weather