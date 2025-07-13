#pragma once

#include "../common.h"


namespace xt::transforms::text {

    /**
     * @struct BertInput
     * @brief Holds the tokenized output ready for a BERT-like model.
     */
    struct BertInput {
        std::vector<long> token_ids;      // The sequence of token IDs.
        std::vector<long> attention_mask; // Mask to avoid performing attention on padding tokens.
        // std::vector<long> token_type_ids; // Segment IDs (usually 0s for single sentence).
    };


    /**
     * @class BertTokenizer
     * @brief A tokenizer that replicates the behavior of BERT's WordPiece tokenizer.
     *
     * This class handles the full tokenization pipeline: normalization,
     * pre-tokenization, WordPiece sub-word splitting, ID conversion, and
     * padding/truncation to a fixed length.
     */
    class BertTokenizer : public xt::Module {
    public:
        /**
         * @brief Constructs the BertTokenizer.
         * @param vocab_path Path to the vocabulary file (e.g., "vocab.txt").
         * @param max_seq_len The fixed sequence length for padding/truncation.
         * @param do_lower_case Whether to convert text to lowercase.
         */
        explicit BertTokenizer(
                const std::string& vocab_path,
                int max_seq_len = 128,
                bool do_lower_case = true
        );

        /**
         * @brief Executes the full tokenization pipeline.
         * @param tensors An initializer list expected to contain a single std::string.
         * @return An std::any containing a `BertInput` struct with the token IDs
         *         and attention mask.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        /**
         * @brief Converts a sequence of tokens into a sequence of IDs.
         * @param tokens The vector of string tokens.
         * @return The vector of corresponding integer IDs.
         */
        auto convert_tokens_to_ids(const std::vector<std::string>& tokens) const -> std::vector<long>;

    private:
        /**
         * @brief Loads the vocabulary file into memory.
         * @param vocab_path Path to the vocabulary file.
         */
        void load_vocab(const std::string& vocab_path);

        /**
         * @brief Performs WordPiece tokenization on a single pre-tokenized word.
         * @param text The word to tokenize.
         * @return A vector of WordPiece tokens.
         */
        auto tokenize_wordpiece(const std::string& text) const -> std::vector<std::string>;

        // Member variables
        std::unordered_map<std::string, long> vocab_;
        int max_seq_len_;
        bool do_lower_case_;

        // Special tokens
        std::string unk_token_ = "[UNK]";
        std::string cls_token_ = "[CLS]";
        std::string sep_token_ = "[SEP]";
        std::string pad_token_ = "[PAD]";
    };

} // namespace xt::transforms::text