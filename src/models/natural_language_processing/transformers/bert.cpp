#include <models/natural_language_processing/transformers/bert.h>


using namespace std;

//
// #include <torch/torch.h>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <map>
// #include <random>
// #include <sstream>
//
// // Vocabulary Class
// class Vocabulary {
// public:
//     Vocabulary(const std::string& vocab_file) {
//         std::ifstream file(vocab_file);
//         std::string token;
//         int idx = 0;
//         while (std::getline(file, token)) {
//             token_to_idx_[token] = idx;
//             idx_to_token_[idx] = token;
//             idx++;
//         }
//         vocab_size_ = idx;
//     }
//
//     int token_to_idx(const std::string& token) const {
//         auto it = token_to_idx_.find(token);
//         return it != token_to_idx_.end() ? it->second : token_to_idx_.at("<unk>");
//     }
//
//     std::string idx_to_token(int idx) const {
//         auto it = idx_to_token_.find(idx);
//         return it != idx_to_token_.end() ? it->second : "<unk>";
//     }
//
//     int vocab_size() const { return vocab_size_; }
//
// private:
//     std::map<std::string, int> token_to_idx_;
//     std::map<int, std::string> idx_to_token_;
//     int vocab_size_;
// };
//
// // Sentence Pair Dataset with MLM
// class SentencePairDataset : public torch::data::Dataset<SentencePairDataset> {
// public:
//     SentencePairDataset(const std::string& sentence_dir, const Vocabulary& vocab, int max_seq_length, float mask_prob = 0.15)
//         : vocab_(vocab), max_seq_length_(max_seq_length), mask_prob_(mask_prob) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//
//         for (const auto& entry : std::filesystem::directory_iterator(sentence_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 sentence_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(sentence_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + sentence_files_[index]);
//         }
//         std::string sentence_a, sentence_b;
//         std::getline(file, sentence_a);
//         std::getline(file, sentence_b);
//         int is_next = (index % 2 == 0) ? 1 : 0; // Alternate next/not-next for NSP
//
//         // Tokenize sentences
//         std::vector<std::string> tokens = {"<cls>"};
//         std::istringstream iss_a(sentence_a);
//         std::string token;
//         while (iss_a >> token && tokens.size() < max_seq_length_ / 2 - 1) {
//             tokens.push_back(token);
//         }
//         tokens.push_back("<sep>");
//         std::istringstream iss_b(sentence_b);
//         while (iss_b >> token && tokens.size() < max_seq_length_ - 2) {
//             tokens.push_back(token);
//         }
//         tokens.push_back("<sep>");
//         while (tokens.size() < max_seq_length_) {
//             tokens.push_back("<pad>");
//         }
//
//         // Create input, MLM target, and mask
//         std::vector<int64_t> input_indices, mlm_target;
//         std::vector<int64_t> masked_positions;
//         std::uniform_real_distribution<float> dist(0.0, 1.0);
//         for (size_t i = 0; i < tokens.size(); ++i) {
//             if (tokens[i] == "<cls>" || tokens[i] == "<sep>" || tokens[i] == "<pad>") {
//                 input_indices.push_back(vocab_.token_to_idx(tokens[i]));
//                 mlm_target.push_back(-100); // Ignore for MLM loss
//                 continue;
//             }
//             if (dist(rng_) < mask_prob_) {
//                 masked_positions.push_back(i);
//                 if (dist(rng_) < 0.8) {
//                     input_indices.push_back(vocab_.token_to_idx("<mask>"));
//                 } else if (dist(rng_) < 0.5) {
//                     input_indices.push_back(vocab_.token_to_idx(tokens[rand() % tokens.size()]));
//                 } else {
//                     input_indices.push_back(vocab_.token_to_idx(tokens[i]));
//                 }
//                 mlm_target.push_back(vocab_.token_to_idx(tokens[i]));
//             } else {
//                 input_indices.push_back(vocab_.token_to_idx(tokens[i]));
//                 mlm_target.push_back(-100);
//             }
//         }
//
//         torch::Tensor input_tensor = torch::tensor(input_indices, torch::kInt64);
//         torch::Tensor mlm_target_tensor = torch::tensor(mlm_target, torch::kInt64);
//         torch::Tensor nsp_label = torch::tensor(is_next, torch::kInt64);
//         torch::Tensor attention_mask = (input_tensor != vocab_.token_to_idx("<pad>")).to(torch::kFloat32);
//
//         return {torch::cat({input_tensor.unsqueeze(0), mlm_target_tensor.unsqueeze(0), nsp_label.unsqueeze(0), attention_mask.unsqueeze(0)}, 0),
//                 torch::Tensor()};
//     }
//
//     torch::optional<size_t> size() const override {
//         return sentence_files_.size();
//     }
//
// private:
//     std::vector<std::string> sentence_files_;
//     const Vocabulary& vocab_;
//     int max_seq_length_;
//     float mask_prob_;
//     std::mt19937 rng_;
// };
//
// // Transformer Encoder Layer
// struct TransformerEncoderLayerImpl : torch::nn::Module {
//     TransformerEncoderLayerImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}, torch::Tensor src_key_padding_mask = {}) {
//         src = norm1->forward(src);
//         auto attn_out = self_attn->forward(src, src, src, {}, src_mask, src_key_padding_mask);
//         src = src + dropout1->forward(std::get<0>(attn_out));
//         src = norm2->forward(src);
//         auto ff_out = linear2->forward(torch::gelu(linear1->forward(src)));
//         src = src + dropout2->forward(ff_out);
//         return src;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerEncoderLayer);
//
// // Transformer Encoder
// struct TransformerEncoderImpl : torch::nn::Module {
//     TransformerEncoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout));
//             register_module("layer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}, torch::Tensor src_key_padding_mask = {}) {
//         torch::Tensor output = src;
//         for (auto& layer : *layers) {
//             output = layer->forward(output, src_mask, src_key_padding_mask);
//         }
//         return output;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(TransformerEncoder);
//
// // Simplified BERT-like Model
// struct BERTImpl : torch::nn::Module {
//     BERTImpl(int vocab_size, int d_model = 512, int nhead = 8, int num_layers = 6, int dim_feedforward = 2048, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         position_embedding = register_module("position_embedding", torch::nn::Embedding(512, d_model));
//         segment_embedding = register_module("segment_embedding", torch::nn::Embedding(2, d_model));
//         encoder = register_module("encoder", TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout));
//         mlm_head = register_module("mlm_head", torch::nn::Linear(d_model, vocab_size));
//         nsp_head = register_module("nsp_head", torch::nn::Linear(d_model, 2));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout));
//         norm = register_module("norm", torch::nn::LayerNorm(d_model));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input_ids, torch::Tensor attention_mask, torch::Tensor segment_ids) {
//         auto batch_size = input_ids.size(0);
//         auto seq_len = input_ids.size(1);
//         auto positions = torch::arange(seq_len, input_ids.device()).unsqueeze(0).expand({batch_size, seq_len});
//
//         auto embeddings = token_embedding->forward(input_ids) +
//                           position_embedding->forward(positions) +
//                           segment_embedding->forward(segment_ids);
//         embeddings = norm->forward(dropout->forward(embeddings));
//
//         auto encoded = encoder->forward(embeddings, {}, 1.0 - attention_mask); // Invert mask for padding
//         auto mlm_logits = mlm_head->forward(encoded); // [batch, seq_len, vocab_size]
//         auto cls_output = encoded.index_select(1, torch::tensor({0}, input_ids.device())).squeeze(1); // CLS token
//         auto nsp_logits = nsp_head->forward(cls_output); // [batch, 2]
//
//         return {mlm_logits, nsp_logits};
//     }
//
//     torch::nn::Embedding token_embedding{nullptr}, position_embedding{nullptr}, segment_embedding{nullptr};
//     TransformerEncoder encoder{nullptr};
//     torch::nn::Linear mlm_head{nullptr}, nsp_head{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(BERT);
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int batch_size = 16;
//         const int num_epochs = 10;
//         const float learning_rate = 0.00005;
//         const int d_model = 512;
//         const int nhead = 8;
//         const int num_layers = 6;
//         const int dim_feedforward = 2048;
//         const float dropout = 0.1;
//         const int max_seq_length = 128;
//         const float mask_prob = 0.15;
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         BERT model(vocab.vocab_size(), d_model, nhead, num_layers, dim_feedforward, dropout);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(0.01));
//
//         // Dataset and DataLoader
//         auto dataset = SentencePairDataset("./data/sentences", vocab, max_seq_length, mask_prob)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             std::move(dataset),
//             torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//             for (auto& batch : *data_loader) {
//                 auto data = batch.data.to(device);
//                 auto input_ids = data[0].to(torch::kInt64);
//                 auto mlm_target = data[1].to(torch::kInt64);
//                 auto nsp_label = data[2].to(torch::kInt64);
//                 auto attention_mask = data[3].to(torch::kFloat32);
//
//                 // Segment IDs (0 for first sentence, 1 for second)
//                 auto segment_ids = torch::zeros_like(input_ids);
//                 auto sep_positions = (input_ids == vocab.token_to_idx("<sep>")).nonzero();
//                 for (int i = 0; i < batch_size; ++i) {
//                     auto sep_idx = sep_positions[i][1].item<int64_t>();
//                     segment_ids[i].slice(0, sep_idx + 1, max_seq_length) = 1;
//                 }
//
//                 optimizer.zero_grad();
//                 auto [mlm_logits, nsp_logits] = model->forward(input_ids, attention_mask, segment_ids);
//
//                 // MLM loss
//                 auto mlm_mask = (mlm_target != -100);
//                 auto mlm_loss = torch::nn::functional::cross_entropy(
//                     mlm_logits.masked_select(mlm_mask.unsqueeze(-1).expand_as(mlm_logits)).view({-1, vocab.vocab_size()}),
//                     mlm_target.masked_select(mlm_mask)
//                 );
//
//                 // NSP loss
//                 auto nsp_loss = torch::nn::functional::cross_entropy(nsp_logits, nsp_label);
//
//                 // Combined loss
//                 auto loss = mlm_loss + nsp_loss;
//                 loss.backward();
//                 torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // Gradient clipping
//                 optimizer.step();
//
//                 total_loss += loss.item<float>();
//                 batch_count++;
//             }
//
//             float avg_loss = total_loss / batch_count;
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << avg_loss << std::endl;
//
//             // Save checkpoint every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "bert_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: bert_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "bert_final.pt");
//         std::cout << "Saved final model: bert_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }



namespace xt::models
{
    BERT::BERT(int num_classes, int in_channels)
    {
    }

    BERT::BERT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void BERT::reset()
    {
    }

    auto BERT::forward(std::initializer_list<std::any> tensors) -> std::any
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
