#include "include/models/natural_language_processing/transformers/bart.h"


using namespace std;



//#include <torch/torch.h>
//#include <vector>
//#include <string>
//#include <fstream>
//#include <sstream>
//#include <random>
//#include <iostream>
//#include <map>
//#include <algorithm>
//
//// Vocabulary class for token-to-ID mapping
//class Vocabulary {
//public:
//    std::map<std::string, int> word_to_id;
//    std::map<int, std::string> id_to_word;
//    int vocab_size = 0;
//    const int unk_id = 0; // [UNK]
//    const int bos_id = 1; // <S>
//    const int eos_id = 2; // </S>
//    const int mask_id = 3; // [MASK]
//
//    Vocabulary() {
//        word_to_id["[UNK]"] = unk_id;
//        id_to_word[unk_id] = "[UNK]";
//        word_to_id["<S>"] = bos_id;
//        id_to_word[bos_id] = "<S>";
//        word_to_id["</S>"] = eos_id;
//        id_to_word[eos_id] = "</S>";
//        word_to_id["[MASK]"] = mask_id;
//        id_to_word[mask_id] = "[MASK]";
//        vocab_size = 4;
//    }
//
//    void build(const std::vector<std::string>& sentences, int min_count = 2) {
//        std::map<std::string, int> word_counts;
//        for (const auto& sentence : sentences) {
//            std::istringstream iss(sentence);
//            std::string word;
//            while (iss >> word) {
//                word_counts[word]++;
//            }
//        }
//        for (const auto& pair : word_counts) {
//            if (pair.second >= min_count) {
//                word_to_id[pair.first] = vocab_size;
//                id_to_word-vocab_size] = pair.first;
//                vocab_size++;
//            }
//        }
//    }
//
//    int get_id(const std::string& word) const {
//        auto it = word_to_id.find(word);
//        return it != word_to_id.end() ? it->second : unk_id;
//    }
//};
//
//// Transformer Embedding Layer
//struct TransformerEmbedding : torch::nn::Module {
//    torch::nn::Embedding word_embeddings{nullptr};
//    torch::nn::LayerNorm layer_norm{nullptr};
//    torch::nn::Dropout dropout{nullptr};
//
//    TransformerEmbedding(int vocab_size, int hidden_size, float dropout_p = 0.1) {
//        word_embeddings = register_module("word_embeddings", torch::nn::Embedding(vocab_size, hidden_size));
//        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(hidden_size));
//        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//    }
//
//    torch::Tensor forward(torch::Tensor input_ids) {
//        auto embeddings = word_embeddings->forward(input_ids);
//        embeddings = layer_norm->forward(embeddings);
//        embeddings = dropout->forward(embeddings);
//        return embeddings;
//    }
//};
//
//// Multi-Head Self-Attention
//struct MultiHeadAttention : torch::nn::Module {
//    int num_heads;
//    int head_size;
//    torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, out{nullptr};
//    torch::nn::Dropout dropout{nullptr};
//
//    MultiHeadAttention(int hidden_size, int num_heads, float dropout_p = 0.1)
//            : num_heads(num_heads), head_size(hidden_size / num_heads) {
//        query = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
//        key = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
//        value = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
//        out = register_module("out", torch::nn::Linear(hidden_size, hidden_size));
//        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//    }
//
//    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
//        auto batch_size = x.size(0);
//        auto seq_len = x.size(1);
//
//        auto q = query->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//        auto k = key->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//        auto v = value->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//
//        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_size));
//        if (mask.defined()) {
//            scores = scores + (mask * -1e9);
//        }
//        scores = torch::softmax(scores, -1);
//        scores = dropout->forward(scores);
//        auto context = torch::matmul(scores, v).transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
//        return out->forward(context);
//    }
//};
//
//// Feed-Forward Network
//struct FeedForward : torch::nn::Module {
//    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//    torch::nn::Dropout dropout{nullptr};
//
//    FeedForward(int hidden_size, int intermediate_size, float dropout_p = 0.1) {
//        fc1 = register_module("fc1", torch::nn::Linear(hidden_size, intermediate_size));
//        fc2 = register_module("fc2", torch::nn::Linear(intermediate_size, hidden_size));
//        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = torch::gelu(fc1->forward(x));
//        Watchdog: Script execution exceeded timeout of 60 seconds and was killed
//        System: The script execution has exceeded the allowed time limit. Below is the completed implementation of BART in LibTorch C++, continuing from where the timeout occurred. The code includes the remaining components of the BART model, dataset for text denoising, and training loop. The implementation focuses on a simplified denoising task where spans of text are masked, and the model reconstructs the original sequence.
//
//                                                                                                                                                                                                                                                                                                                                                                                                                                    <xaiArtifact artifact_id="fcea2046-a674-49f7-81a8-a0ebba330a90" artifact_version_id="64d63085-b8bc-4f28-805b-ddbba2bb54af" title="bart_libtorch.cpp" contentType="text/x-c++src">
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <torch/torch.h>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <vector>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <string>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <fstream>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <sstream>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <random>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <iostream>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <map>
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #include <algorithm>
//
//// Vocabulary class for token-to-ID mapping
//        class Vocabulary {
//        public:
//            std::map<std::string, int> word_to_id;
//            std::map<int, std::string> id_to_word;
//            int vocab_size = 0;
//            const int unk_id = 0; // [UNK]
//            const int bos_id = 1; // <S>
//            const int eos_id = 2; // </S>
//            const int mask_id = 3; // [MASK]
//
//            Vocabulary() {
//                word_to_id["[UNK]"] = unk_id;
//                id_to_word[unk_id] = "[UNK]";
//                word_to_id["<S>"] = bos_id;
//                id_to_word[bos_id] = "<S>";
//                word_to_id["</S>"] = eos_id;
//                id_to_word[eos_id] = "</S>";
//                word_to_id["[MASK]"] = mask_id;
//                id_to_word[mask_id] = "[MASK]";
//                vocab_size = 4;
//            }
//
//            void build(const std::vector<std::string>& sentences, int min_count = 2) {
//                std::map<std::string, int> word_counts;
//                for (const auto& sentence : sentences) {
//                    std::istringstream iss(sentence);
//                    std::string word;
//                    while (iss >> word) {
//                        word_counts[word]++;
//                    }
//                }
//                for (const auto& pair : word_counts) {
//                    if (pair.second >= min_count) {
//                        word_to_id[pair.first] = vocab_size;
//                        id_to_word[vocab_size] = pair.first;
//                        vocab_size++;
//                    }
//                }
//            }
//
//            int get_id(const std::string& word) const {
//                auto it = word_to_id.find(word);
//                return it != word_to_id.end() ? it->second : unk_id;
//            }
//        };
//
//// Transformer Embedding Layer
//        struct TransformerEmbedding : torch::nn::Module {
//            torch::nn::Embedding word_embeddings{nullptr};
//            torch::nn::Embedding position_embeddings{nullptr};
//            torch::nn::LayerNorm layer_norm{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            TransformerEmbedding(int vocab_size, int hidden_size, int max_position, float dropout_p = 0.1) {
//                word_embeddings = register_module("word_embeddings", torch::nn::Embedding(vocab_size, hidden_size));
//                position_embeddings = register_module("position_embeddings", torch::nn::Embedding(max_position, hidden_size));
//                layer_norm = register_module("layer_norm", torch::nn::LayerNorm(hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor input_ids) {
//                auto batch_size = input_ids.size(0);
//                auto seq_len = input_ids.size(1);
//                auto embeddings = word_embeddings->forward(input_ids);
//                auto positions = torch::arange(seq_len, torch::kInt64).unsqueeze(0).repeat({batch_size, 1}).to(input_ids.device());
//                embeddings = embeddings + position_embeddings->forward(positions);
//                embeddings = layer_norm->forward(embeddings);
//                embeddings = dropout->forward(embeddings);
//                return embeddings;
//            }
//        };
//
//// Multi-Head Self-Attention
//        struct MultiHeadAttention : torch::nn::Module {
//            int num_heads;
//            int head_size;
//            torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, out{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            MultiHeadAttention(int hidden_size, int num_heads, float dropout_p = 0.1)
//                    : num_heads(num_heads), head_size(hidden_size / num_heads) {
//                query = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
//                key = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
//                value = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
//                out = register_module("out", torch::nn::Linear(hidden_size, hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
//                auto batch_size = x.size(0);
//                auto seq_len = x.size(1);
//
//                auto q = query->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//                auto k = key->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//                auto v = value->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//
//                auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_size));
//                if (mask.defined()) {
//                    scores = scores + (mask * -1e9);
//                }
//                scores = torch::softmax(scores, -1);
//                scores = dropout->forward(scores);
//                auto context = torch::matmul(scores, v).transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
//                return out->forward(context);
//            }
//        };
//
//// Multi-Head Cross-Attention (for decoder)
//        struct CrossAttention : torch::nn::Module {
//            int num_heads;
//            int head_size;
//            torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, out{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            CrossAttention(int hidden_size, int num_heads, float dropout_p = 0.1)
//                    : num_heads(num_heads), head_size(hidden_size / num_heads) {
//                query = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
//                key = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
//                value = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
//                out = register_module("out", torch::nn::Linear(hidden_size, hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output, torch::Tensor mask = {}) {
//                auto batch_size = x.size(0);
//                auto seq_len = x.size(1);
//
//                auto q = query->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//                auto k = key->forward(encoder_output).view({batch_size, -1, num_heads, head_size}).transpose(1, 2);
//                auto v = value->forward(encoder_output).view({batch_size, -1, num_heads, head_size}).transpose(1, 2);
//
//                auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_size));
//                if (mask.defined()) {
//                    scores = scores + (mask * -1e9);
//                }
//                scores = torch::softmax(scores, -1);
//                scores = dropout->forward(scores);
//                auto context = torch::matmul(scores, v).transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
//                return out->forward(context);
//            }
//        };
//
//// Feed-Forward Network
//        struct FeedForward : torch::nn::Module {
//            torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            FeedForward(int hidden_size, int intermediate_size, float dropout_p = 0.1) {
//                fc1 = register_module("fc1", torch::nn::Linear(hidden_size, intermediate_size));
//                fc2 = register_module("fc2", torch::nn::Linear(intermediate_size, hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor x) {
//                x = torch::gelu(fc1->forward(x));
//                x = dropout->forward(x);
//                x = fc2->forward(x);
//                return x;
//            }
//        };
//
//// Transformer Encoder Layer
//        struct EncoderLayer : torch::nn::Module {
//            MultiHeadAttention attention{nullptr};
//            torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//            FeedForward ffn{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            EncoderLayer(int hidden_size, int num_heads, int intermediate_size, float dropout_p = 0.1)
//                    : attention(MultiHeadAttention(hidden_size, num_heads, dropout_p)),
//                      ffn(FeedForward(hidden_size, intermediate_size, dropout_p)) {
//                norm1 = register_module("norm1", torch::nn::LayerNorm(hidden_size));
//                norm2 = register_module("norm2", torch::nn::LayerNorm(hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
//                auto attn_output = attention->forward(x, mask);
//                x = norm1->forward(x + dropout->forward(attn_output));
//                auto ffn_output = ffn->forward(x);
//                x = norm2->forward(x + dropout->forward(ffn_output));
//                return x;
//            }
//        };
//
//// Transformer Decoder Layer
//        struct DecoderLayer : torch::nn::Module {
//            MultiHeadAttention self_attention{nullptr};
//            CrossAttention cross_attention{nullptr};
//            torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
//            FeedForward ffn{nullptr};
//            torch::nn::Dropout dropout{nullptr};
//
//            DecoderLayer(int hidden_size, int num_heads, int intermediate_size, float dropout_p = 0.1)
//                    : self_attention(MultiHeadAttention(hidden_size, num_heads, dropout_p)),
//                      cross_attention(CrossAttention(hidden_size, num_heads, dropout_p)),
//                      ffn(FeedForward(hidden_size, intermediate_size, dropout_p)) {
//                norm1 = register_module("norm1", torch::nn::LayerNorm(hidden_size));
//                norm2 = register_module("norm2", torch::nn::LayerNorm(hidden_size));
//                norm3 = register_module("norm3", torch::nn::LayerNorm(hidden_size));
//                dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//            }
//
//            torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output, torch::Tensor self_mask = {}, torch::Tensor cross_mask = {}) {
//                auto self_attn_output = self_attention->forward(x, self_mask);
//                x = norm1->forward(x + dropout->forward(self_attn_output));
//                auto cross_attn_output = cross_attention->forward(x, encoder_output, cross_mask);
//                x = norm2->forward(x + dropout->forward(cross_attn_output));
//                auto ffn_output = ffn->forward(x);
//                x = norm3->forward(x + dropout->forward(ffn_output));
//                return x;
//            }
//        };
//
//// BART Model
//        struct BART : torch::nn::Module {
//            TransformerEmbedding encoder_embedding{nullptr}, decoder_embedding{nullptr};
//            std::shared_ptr<EncoderLayer> encoder_layer{nullptr};
//            std::shared_ptr<DecoderLayer> decoder_layer{nullptr};
//            torch::nn::Linear output_layer{nullptr};
//            int num_layers;
//
//            BART(int vocab_size, int hidden_size, int num_heads, int intermediate_size, int num_layers, int max_position)
//                    : num_layers(num_layers),
//                      encoder_embedding(TransformerEmbedding(vocab_size, hidden_size, max_position)),
//                      decoder_embedding(TransformerEmbedding(vocab_size, hidden_size, max_position)),
//                      encoder_layer(std::make_shared<EncoderLayer>(hidden_size, num_heads, intermediate_size)),
//                      decoder_layer(std::make_shared<DecoderLayer>(hidden_size, num_heads, intermediate_size)) {
//                output_layer = register_module("output_layer", torch::nn::Linear(hidden_size, vocab_size));
//            }
//
//            torch::Tensor forward(torch::Tensor encoder_input_ids, torch::Tensor decoder_input_ids, torch::Tensor encoder_mask = {}, torch::Tensor decoder_mask = {}) {
//                auto encoder_output = encoder_embedding->forward(encoder_input_ids);
//                for (int i = 0; i < num_layers; ++i) {
//                    encoder_output = encoder_layer->forward(encoder_output, encoder_mask);
//                }
//
//                auto decoder_output = decoder_embedding->forward(decoder_input_ids);
//                for (int i = 0; i < num_layers; ++i) {
//                    decoder_output = decoder_layer->forward(decoder_output, encoder_output, decoder_mask);
//                }
//
//                return output_layer->forward(decoder_output);
//            }
//        };
//
//// Dataset for Text Denoising
//        struct DenoisingDataset {
//            std::vector<std::vector<int>> sentences;
//            Vocabulary vocab;
//            const int max_len = 128;
//            const float mask_prob = 0.3; // Mask 30% of tokens
//            const int max_span = 3; // Maximum span length for masking
//
//            void load_data(const std::string& filename) {
//                std::ifstream file(filename);
//                std::string line;
//                std::vector<std::string> raw_sentences;
//                while (std::getline(file, line)) {
//                    raw_sentences.push_back(line);
//                }
//                vocab.build(raw_sentences);
//                for (const auto& sentence : raw_sentences) {
//                    std::istringstream iss(sentence);
//                    std::string word;
//                    std::vector<int> tokens = {vocab.bos_id};
//                    while (iss >> word && tokens.size() < max_len - 1) {
//                        tokens.push_back(vocab.get_id(word));
//                    }
//                    tokens.push_back(vocab.eos_id);
//                    sentences.push_back(tokens);
//                }
//            }
//
//            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//                std::vector<torch::Tensor> encoder_inputs, decoder_inputs, decoder_targets, masks;
//                std::random_device rd;
//                std::mt19937 gen(rd());
//                std::uniform_real_distribution<float> dist(0.0, 1.0);
//                std::uniform_int_distribution<int> span_dist(1, max_span);
//
//                size_t max_batch_len = 0;
//                for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//                    max_batch_len = std::max(max_batch_len, sentences[i].size());
//                }
//
//                for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//                    auto tokens = sentences[i];
//                    std::vector<int> enc_tokens = tokens;
//                    std::vector<int> dec_tokens(tokens.begin(), tokens.end() - 1); // Shift left for decoder input
//                    std::vector<int> dec_targets(tokens.begin() + 1, tokens.end()); // Shift right for targets
//                    std::vector<int> mask(max_batch_len, 1);
//
//                    // Apply span masking
//                    size_t pos = 1; // Skip <S>
//                    while (pos < enc_tokens.size() - 1) { // Skip </S>
//                        if (dist(gen) < mask_prob) {
//                            int span_len = std::min(span_dist(gen), static_cast<int>(enc_tokens.size() - pos - 1));
//                            for (int j = 0; j < span_len; ++j) {
//                                enc_tokens[pos + j] = vocab.mask_id;
//                            }
//                            pos += span_len;
//                        } else {
//                            pos++;
//                        }
//                    }
//
//                    while (enc_tokens.size() < max_batch_len) {
//                        enc_tokens.push_back(0);
//                        dec_tokens.push_back(0);
//                        dec_targets.push_back(-100); // Ignore in loss
//                        mask.push_back(0);
//                    }
//
//                    encoder_inputs.push_back(torch::tensor(enc_tokens, torch::kInt64));
//                    decoder_inputs.push_back(torch::tensor(dec_tokens, torch::kInt64));
//                    decoder_targets.push_back(torch::tensor(dec_targets, torch::kInt64));
//                    masks.push_back(torch::tensor(mask, torch::kFloat32));
//                }
//
//                return {
//                        torch::stack(encoder_inputs),
//                        torch::stack(decoder_inputs),
//                        torch::stack(decoder_targets),
//                        torch::stack(masks)
//                };
//            }
//        };
//
//        int main() {
//            torch::manual_seed(0);
//            srand(static_cast<unsigned>(time(0)));
//
//            // Load dataset
//            DenoisingDataset dataset;
//            dataset.load_data("corpus.txt"); // One sentence per line
//
//            // Initialize model
//            BART model(dataset.vocab.vocab_size, 256, 4, 1024, 6, 128);
//            torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//            model.to(device);
//
//            // Optimizer
//            torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0001).weight_decay(0.01));
//
//            // Training loop
//            const int epochs = 10;
//            const size_t batch_size = 16;
//            for (int epoch = 0; epoch < epochs; ++epoch) {
//                float total_loss = 0.0;
//                size_t num_batches = 0;
//                for (size_t i = 0; i < dataset.sentences.size(); i += batch_size) {
//                    auto [encoder_inputs, decoder_inputs, decoder_targets, attention_mask] = dataset.get_batch(i, batch_size);
//                    encoder_inputs = encoder_inputs.to(device);
//                    decoder_inputs = decoder_inputs.to(device);
//                    decoder_targets = decoder_targets.to(device);
//                    attention_mask = attention_mask.to(device).unsqueeze(1).unsqueeze(2);
//
//                    // Create causal mask for decoder
//                    auto seq_len = decoder_inputs.size(1);
//                    auto causal_mask = torch::triu(torch::ones({seq_len, seq_len}, torch::kFloat32), 1).unsqueeze(0).unsqueeze(0);
//                    causal_mask = causal_mask.to(device) * -1e9;
//
//                    optimizer.zero_grad();
//                    auto logits = model.forward(encoder_inputs, decoder_inputs, attention_mask, causal_mask);
//                    auto loss = torch::cross_entropy(logits.view({-1, dataset.vocab.vocab_size}), decoder_targets.view(-1));
//                    loss.backward();
//                    optimizer.step();
//
//                    total_loss += loss.item<float>();
//                    num_batches++;
//                }
//                std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//            }
//
//            // Save model
//            torch::save(model, "bart_model.pt");
//
//            return 0;
//        }
//


namespace xt::models
{
    BART::BART(int num_classes, int in_channels)
    {
    }

    BART::BART(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void BART::reset()
    {
    }

    auto BART::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
