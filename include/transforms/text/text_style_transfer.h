#pragma once

#include "../common.h"
#include <string>
#include <memory> // For std::shared_ptr

namespace xt::transforms::text {

    /**
     * @class StyleTransferClient
     * @brief An abstract interface for a text style transfer model/service.
     *
     * This contract allows the TextStyleTransfer transform to remain independent
     * of the specific model implementation (e.g., local ONNX model, remote API).
     */
    class StyleTransferClient {
    public:
        virtual ~StyleTransferClient() = default;

        /**
         * @brief Transfers the style of a given text.
         * @param text The source text to transform.
         * @param target_style A string identifier for the desired output style
         *                     (e.g., "formal", "informal", "shakespearean").
         * @return The text rewritten in the target style.
         */
        virtual auto transfer(const std::string& text, const std::string& target_style) const -> std::string = 0;
    };


    /**
     * @class TextStyleTransfer
     * @brief A text transformation that rewrites an input string to a specified style.
     *
     * This transform relies on an underlying client that provides the actual
     * style transfer capabilities from a pre-trained model.
     */
    class TextStyleTransfer : public xt::Module {
    public:
        /**
         * @brief Constructs the TextStyleTransfer transform.
         *
         * @param client A shared pointer to a concrete StyleTransferClient.
         * @param target_style A string representing the style to convert to. This
         *                     must be a style understood by the provided client.
         */
        explicit TextStyleTransfer(
                std::shared_ptr<StyleTransferClient> client,
                const std::string& target_style
        );

        /**
         * @brief Executes the style transfer operation.
         * @param tensors An initializer list expected to contain a single std::string.
         * @return An std::any containing the resulting style-transferred std::string.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::shared_ptr<StyleTransferClient> client_;
        std::string target_style_;
    };

} // namespace xt::transforms::text