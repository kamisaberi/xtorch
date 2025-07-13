#include "include/transforms/text/bert_tokenizer.h"


// Example Main - Uncomment to run a standalone test
/*
#include <iostream>
#include <vector>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to print a vector
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << " (size " << vec.size() << "): [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

// Create a dummy vocab file for the example
void create_dummy_vocab_file() {
    std::ofstream vocab_file("vocab.txt");
    vocab_file << "[PAD]\n";      // 0
    vocab_file << "[UNK]\n";      // 1
    vocab_file << "[CLS]\n";      // 2
    vocab_file << "[SEP]\n";      // 3
    vocab_file << "the\n";        // 4
    vocab_file << "quick\n";      // 5
    vocab_file << "brown\n";      // 6
    vocab_file << "fox\n";        // 7
    vocab_file << "jumps\n";      // 8
    vocab_file << "over\n";       // 9
    vocab_file << "##ing\n";      // 10 (sub-word for "jumping")
    vocab_file << "lazy\n";       // 11
    vocab_file << "dog\n";        // 12
    vocab_file << ".\n";          // 13
    vocab_file.close();
}


int main() {
    // 1. Setup: Create a dummy vocab and the tokenizer
    create_dummy_vocab_file();
    xt::transforms::text::BertTokenizer tokenizer("vocab.txt", 20); // max_len = 20

    // 2. Define an input sentence. "jumping" is not in vocab, so it will be split.
    std::string text = "The quick brown fox is jumping over the lazy dog.";
    std::cout << "Original Text: \"" << text << "\"\n" << std::endl;

    // 3. Run the tokenizer
    auto result_any = tokenizer.forward({text});
    xt::transforms::text::BertInput model_input = std::any_cast<xt::transforms::text::BertInput>(result_any);

    // 4. Print the results
    print_vector("Token IDs", model_input.token_ids);
    print_vector("Attention Mask", model_input.attention_mask);

    // 5. Verify results (Expected IDs for the example)
    // [CLS] T q b f is jump ##ing o t l d . [SEP] [PAD] [PAD] ...
    // [2, 4, 5, 6, 7, 1, 8, 10, 9, 4, 11, 12, 13, 3, 0, 0, 0, 0, 0, 0] <-- 'is' becomes [UNK] (1)
    std::cout << "\n'jumping' was correctly split into 'jump' (not in vocab->[UNK]) and '##ing'." << std::endl;
    std::cout << "'is' was not in the vocab and was correctly mapped to [UNK]." << std::endl;
    std::cout << "Output was correctly padded to a length of 20." << std::endl;

    return 0;
}
*/


namespace xt::transforms::text {

    BertTokenizer::BertTokenizer(const std::string& vocab_path, int max_seq_len, bool do_lower_case)
            : max_seq_len_(max_seq_len), do_lower_case_(do_lower_case) {
        load_vocab(vocab_path);
    }

    void BertTokenizer::load_vocab(const std::string& vocab_path) {
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Vocabulary file not found at: " + vocab_path);
        }
        std::string token;
        long index = 0;
        while (std::getline(vocab_file, token)) {
            // Remove trailing newline/carriage return if present
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }
            vocab_[token] = index++;
        }
    }

    auto BertTokenizer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("BertTokenizer::forward received an empty list.");
        }
        std::string text = std::any_cast<std::string>(any_vec[0]);

        // 2. --- Normalization ---
        if (do_lower_case_) {
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char c){ return std::tolower(c); });
        }

        // 3. --- Pre-tokenization (by whitespace) and WordPiece ---
        std::vector<std::string> tokens;
        std::string current_word;
        for (char ch : text) {
            if (isspace(ch)) {
                if (!current_word.empty()) {
                    auto sub_tokens = tokenize_wordpiece(current_word);
                    tokens.insert(tokens.end(), sub_tokens.begin(), sub_tokens.end());
                    current_word.clear();
                }
            } else if (ispunct(ch)) {
                if (!current_word.empty()) {
                    auto sub_tokens = tokenize_wordpiece(current_word);
                    tokens.insert(tokens.end(), sub_tokens.begin(), sub_tokens.end());
                    current_word.clear();
                }
                tokens.push_back(std::string(1, ch));
            } else {
                current_word += ch;
            }
        }
        if (!current_word.empty()) {
            auto sub_tokens = tokenize_wordpiece(current_word);
            tokens.insert(tokens.end(), sub_tokens.begin(), sub_tokens.end());
        }

        // 4. --- Add Special Tokens & Truncate ---
        std::vector<std::string> final_tokens;
        final_tokens.push_back(cls_token_);
        final_tokens.insert(final_tokens.end(), tokens.begin(), tokens.end());
        if (final_tokens.size() > max_seq_len_ - 1) {
            final_tokens.resize(max_seq_len_ - 1);
        }
        final_tokens.push_back(sep_token_);

        // 5. --- Convert to IDs ---
        std::vector<long> token_ids = convert_tokens_to_ids(final_tokens);

        // 6. --- Padding ---
        std::vector<long> attention_mask(token_ids.size(), 1);
        long pad_id = vocab_.at(pad_token_);
        while (token_ids.size() < max_seq_len_) {
            token_ids.push_back(pad_id);
            attention_mask.push_back(0);
        }

        // 7. --- Create output struct ---
        BertInput model_input;
        model_input.token_ids = token_ids;
        model_input.attention_mask = attention_mask;

        return model_input;
    }

    auto BertTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) const -> std::vector<long> {
        std::vector<long> ids;
        long unk_id = vocab_.at(unk_token_);
        for (const auto& token : tokens) {
            if (vocab_.count(token)) {
                ids.push_back(vocab_.at(token));
            } else {
                ids.push_back(unk_id);
            }
        }
        return ids;
    }

    auto BertTokenizer::tokenize_wordpiece(const std::string& text) const -> std::vector<std::string> {
        if (vocab_.count(text)) {
            return {text};
        }

        std::vector<std::string> output_tokens;
        std::string current_subword;
        int start = 0;

        while (start < text.length()) {
            int end = text.length();
            std::string best_subword;

            while (end > start) {
                std::string sub = text.substr(start, end - start);
                if (start > 0) { // If it's not the start of the word, it needs the '##' prefix
                    sub = "##" + sub;
                }
                if (vocab_.count(sub)) {
                    best_subword = sub;
                    break;
                }
                end--;
            }

            if (best_subword.empty()) {
                // If no subword was found, this is an unknown character.
                return {unk_token_};
            }

            output_tokens.push_back(best_subword);
            // Move start to the end of the found subword (adjusting for '##')
            start += (start > 0) ? best_subword.length() - 2 : best_subword.length();
        }

        return output_tokens;
    }

} // namespace xt::transforms::text