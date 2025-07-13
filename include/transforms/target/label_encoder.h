#pragma once

#include "../common.h"

#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class LabelEncoder
     * @brief A target transformation that converts categorical labels (strings)
     *        into unique integer IDs.
     *
     * This encoder is stateful. It builds a vocabulary of labels as it encounters
     * them and assigns a new, incrementing integer ID to each unique label.
     * It will always return the same ID for the same label on subsequent calls.
     */
    class LabelEncoder : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        LabelEncoder();

        /**
         * @brief Executes the label encoding operation.
         *
         * If the label has been seen before, its previously assigned ID is returned.
         * If the label is new, a new ID is assigned, stored, and returned.
         *
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         *                It will be converted to a string for mapping.
         * @return An std::any containing the assigned integer ID as a `long`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        /**
         * @brief Returns a copy of the internal vocabulary map.
         * @return An `std::unordered_map<std::string, long>` of the learned mappings.
         */
        auto get_mapping() const -> std::unordered_map<std::string, long>;

    private:
        std::unordered_map<std::string, long> mapping_;
        long next_id_;
    };

} // namespace xt::transforms::target