#include "include/models/natural_language_processing/rnn/seq2seq.h"


using namespace std;


// #include <torch/torch.h>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <random>
// #include <iostream>
// #include <map>
// #include <algorithm>
//
// // Vocabulary class for token-to-ID mapping
// class Vocabulary {
// public:
//     std::map<std::string, int> word_to_id;
//     std::map<int, std::string> id_to_word;
//     int vocab_size = 0;
//     const int unk_id = 0; // <unk>
//     const int sos_id = 1; // <sos>
//     const int eos_id = 2; // <eos>
//     const int pad_id = 3; // <pad>
//
//     Vocabulary() {
//         word_to_id["<unk>"] = unk_id;
//         id_to_word[unk_id] = "<unk>";
//         word_to_id["<sos>"] = sos_id;
//         id_to_word[sos_id] = "<sos>";
//         word_to_id["<eos>"] = eos_id;
//         id_to_word[eos_id] = "<eos>";
//         word_to_id["<pad>"] = pad_id;
//         id_to_word[pad_id] = "<pad>";
//         vocab_size = 4;
//     }
//
//     void build(const std::vector<std::string>& sentences, int min_count = 1) {
//         std::map<std::string, int> word_counts;
//         for (const auto& sentence : sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             while (iss >> word) {
//                 word_counts[word]++;
//             }
//         }
//         for (const auto& pair : word_counts) {
//             if (pair.second >= min_count) {
//                 word_to_id[pair.first] = vocab_size;
//                 id_to_word[vocab_size] = pair.first;
//                 vocab_size++;
//             }
//         }
//     }
//
//     int get_id(const std::string& word) const {
//         auto it = word_to_id.find(word);
//         return it != word_to_id.end() ? it->second : unk_id;
//     }
// };
//
// // Encoder LSTM
// struct EncoderLSTM : torch::nn::Module {
//     torch::nn::Embedding embedding{nullptr};
//     torch::nn::LSTM lstm{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     EncoderLSTM(int vocab_size, int embedding_size, int hidden_size, int num_layers, float dropout_p = 0.1) {
//         embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embedding_size));
//         lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embedding_size, hidden_size)
//                                                         .num_layers(num_layers)
//                                                         .dropout(dropout_p)
//                                                         .batch_first(true)));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
//         auto embedded = embedding->forward(input);
//         embedded = dropout->forward(embedded);
//         auto [output, states] = lstm->forward(embedded);
//         auto [h, c] = states;
//         return {output, h, c};
//     }
// };
//
// // Attention Mechanism (Bahdanau Attention)
// struct Attention : torch::nn::Module {
//     torch::nn::Linear W_a{nullptr}, U_a{nullptr}, V_a{nullptr};
//
//     Attention(int hidden_size) {
//         W_a = register_module("W_a", torch::nn::Linear(hidden_size, hidden_size));
//         U_a = register_module("U_a", torch::nn::Linear(hidden_size, hidden_size));
//         V_a = register_module("V_a", torch::nn::Linear(hidden_size, 1));
//     }
//
//     torch::Tensor forward(torch::Tensor decoder_hidden, torch::Tensor encoder_outputs) {
//         auto batch_size = encoder_outputs.size(0);
//         auto src_len = encoder_outputs.size(1);
//
//         // Repeat decoder hidden state src_len times
//         auto hidden = decoder_hidden.unsqueeze(2).repeat({1, 1, src_len, 1});
//         auto energy = torch::tanh(W_a->forward(encoder_outputs) + U_a->forward(hidden.squeeze(3)));
//         auto attention = V_a->forward(energy).squeeze(-1);
//         attention = torch::softmax(attention, 2);
//         return attention;
//     }
// };
//
// // Decoder LSTM with Attention
// struct DecoderLSTM : torch::nn::Module {
//     torch::nn::Embedding embedding{nullptr};
//     torch::nn::LSTM lstm{nullptr};
//     torch::nn::Linear out{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     Attention attention{nullptr};
//
//     DecoderLSTM(int vocab_size, int embedding_size, int hidden_size, int num_layers, float dropout_p = 0.1)
//         : attention(Attention(hidden_size)) {
//         embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embedding_size));
//         lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embedding_size + hidden_size, hidden_size)
//                                                         .num_layers(num_layers)
//                                                         .dropout(dropout_p)
//                                                         .batch_first(true)));
//         out = register_module("out", torch::nn::Linear(hidden_size, vocab_size));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
//         torch::Tensor input, torch::Tensor h, torch::Tensor c, torch::Tensor encoder_outputs) {
//         auto embedded = embedding->forward(input);
//         embedded = dropout->forward(embedded);
//
//         // Apply attention
//         auto attn_weights = attention->forward(h[-1], encoder_outputs);
//         auto context = torch::bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1).unsqueeze(1);
//
//         // Concatenate context and embedded input
//         auto lstm_input = torch::cat({embedded, context}, -1);
//         auto [output, states] = lstm->forward(lstm_input);
//         auto [h_new, c_new] = states;
//
//         output = output.squeeze(1);
//         output = out->forward(output);
//         return {output, h_new, c_new};
//     }
// };
//
// // Seq2Seq Model
// struct Seq2Seq : torch::nn::Module {
//     EncoderLSTM encoder{nullptr};
//     DecoderLSTM decoder{nullptr};
//     int target_vocab_size;
//
//     Seq2Seq(int src_vocab_size, int tgt_vocab_size, int embedding_size, int hidden_size, int num_layers)
//         : target_vocab_size(tgt_vocab_size) {
//         encoder = register_module("encoder", EncoderLSTM(src_vocab_size, embedding_size, hidden_size, num_layers));
//         decoder = register_module("decoder", DecoderLSTM(tgt_vocab_size, embedding_size, hidden_size, num_layers));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encoder_forward(torch::Tensor src) {
//         return encoder->forward(src);
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> decoder_forward(
//         torch::Tensor tgt, torch::Tensor h, torch::Tensor c, torch::Tensor encoder_outputs) {
//         return decoder->forward(tgt, h, c, encoder_outputs);
//     }
// };
//
// // Translation Dataset
// struct TranslationDataset {
//     std::vector<std::pair<std::vector<int>, std::vector<int>>> sentence_pairs;
//     Vocabulary src_vocab, tgt_vocab;
//     const int max_len = 20;
//
//     void load_data(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> src_sentences, tgt_sentences;
//         while (std::getline(file, line)) {
//             std::istringstream iss(line);
//             std::string src, tgt;
//             std::getline(iss, src, '\t');
//             std::getline(iss, tgt);
//             src_sentences.push_back(src);
//             tgt_sentences.push_back(tgt);
//         }
//         src_vocab.build(src_sentences);
//         tgt_vocab.build(tgt_sentences);
//
//         for (size_t i = 0; i < src_sentences.size(); ++i) {
//             std::istringstream src_iss(src_sentences[i]), tgt_iss(tgt_sentences[i]);
//             std::string word;
//             std::vector<int> src_tokens = {src_vocab.sos_id};
//             std::vector<int> tgt_tokens = {tgt_vocab.sos_id};
//             while (src_iss >> word && src_tokens.size() < max_len - 1) {
//                 src_tokens.push_back(src_vocab.get_id(word));
//             }
//             src_tokens.push_back(src_vocab.eos_id);
//             while (tgt_iss >> word && tgt_tokens.size() < max_len - 1) {
//                 tgt_tokens.push_back(tgt_vocab.get_id(word));
//             }
//             tgt_tokens.push_back(tgt_vocab.eos_id);
//             sentence_pairs.emplace_back(src_tokens, tgt_tokens);
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<torch::Tensor> src_batch, tgt_batch;
//         size_t max_src_len = 0, max_tgt_len = 0;
//         for (size_t i = idx; i < std::min(idx + batch_size, sentence_pairs.size()); ++i) {
//             max_src_len = std::max(max_src_len, sentence_pairs[i].first.size());
//             max_tgt_len = std::max(max_tgt_len, sentence_pairs[i].second.size());
//         }
//
//         for (size_t i = idx; i < std::min(idx + batch_size, sentence_pairs[i].size()); ++i) {
//             auto& [src, tgt] = sentence_pairs[i];
//             std::vector<int> src_padded = src, tgt_padded = tgt;
//             while (src_padded.size() < max_src_len) {
//                 src_padded.push_back(src_vocab.pad_id);
//             }
//             while (tgt_padded.size() < max_tgt_len) {
//                 tgt_padded.push_back(tgt_vocab.pad_id);
//             }
//             src_batch.push_back(torch::tensor(src_padded, torch::kInt64));
//             tgt_batch.push_back(torch::tensor(tgt_padded, torch::kInt64));
//         }
//         return {torch::stack(src_batch), torch::stack(tgt_batch)};
//     }
// };
//
// int main() {
//     torch::manual_seed(0);
//     srand(static_cast<unsigned>(time(0)));
//
//     // Load dataset
//     TranslationDataset dataset;
//     dataset.load_data("data.txt"); // Format: English \t French per line
//
//     // Initialize model
//     Seq2Seq model(dataset.src_vocab.vocab_size, dataset.tgt_vocab.vocab_size, 128, 256, 2);
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     model.to(device);
//
//     // Optimizer
//     torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
//
//     // Training loop
//     const int epochs = 20;
//     const size_t batch_size = 32;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(0.0, 1.0);
//     const float teacher_forcing_ratio = 0.5;
//
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         model.train();
//
//         for (size_t i = 0; i < dataset.sentence_pairs.size(); i += batch_size) {
//             auto [src, tgt] = dataset.get_batch(i, batch_size);
//             src = src.to(device);
//             tgt = tgt.to(device);
//
//             optimizer.zero_grad();
//
//             // Encoder forward pass
//             auto [encoder_outputs, h, c] = model.encoder_forward(src);
//
//             // Decoder forward pass
//             auto batch_size_actual = src.size(0);
//             auto tgt_len = tgt.size(1);
//             torch::Tensor outputs = torch::zeros({batch_size_actual, tgt_len - 1, dataset.tgt_vocab.vocab_size}).to(device);
//             torch::Tensor decoder_input = tgt.slice(1, 0, 1); // Start with <sos>
//
//             for (int t = 0; t < tgt_len - 1; ++t) {
//                 auto [output, h_new, c_new] = model.decoder_forward(decoder_input, h, c, encoder_outputs);
//                 outputs.slice(1, t, t + 1) = output.unsqueeze(1);
//                 h = h_new;
//                 c = c_new;
//
//                 // Teacher forcing
//                 bool use_teacher_forcing = dist(gen) < teacher_forcing_ratio;
//                 if (use_teacher_forcing) {
//                     decoder_input = tgt.slice(1, t + 1, t + 2);
//                 } else {
//                     decoder_input = output.argmax(-1).detach();
//                 }
//             }
//
//             // Compute loss
//             auto loss = torch::cross_entropy(outputs.view({-1, dataset.tgt_vocab.vocab_size}), tgt.slice(1, 1, tgt_len).contiguous().view(-1));
//             loss.backward();
//             torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
//             optimizer.step();
//
//             total_loss += loss.item<float>();
//             num_batches++;
//         }
//         std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "seq2seq_model.pt");
//
//     return 0;
// }


namespace xt::models
{
    Seq2Seq::Seq2Seq(int num_classes, int in_channels)
    {
    }

    Seq2Seq::Seq2Seq(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Seq2Seq::reset()
    {
    }

    auto Seq2Seq::forward(std::initializer_list<std::any> tensors) -> std::any
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
