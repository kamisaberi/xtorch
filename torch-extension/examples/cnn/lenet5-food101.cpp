#include "includes/base.h"
#include "../../include/datasets/food.h"
#include "../../include/models/cnn/lenet5.h"
#include "../../include/definitions/transforms.h"

using namespace std;
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

int main() {
    std::vector<int64_t> size = {32, 32};

    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    auto dataset = torch::ext::data::datasets::Food101("/home/kami/Documents/temp/", DataMode::TRAIN, false );

//    auto dataset = torch::ext::data::datasets::Imagenette("/home/kami/Documents/temp/", DataMode::TRAIN, true,torch::ext::data::datasets::ImageType::PX160 );

    auto transformed_dataset = dataset
            .map(torch::ext::data::transforms::resize(size))
            .map(torch::ext::data::transforms::normalize(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    torch::ext::models::LeNet5 model(10, 3 );
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;
        auto train_loader_iterator = train_loader->begin();
        auto train_loader_end = train_loader->end();
        while (train_loader_iterator != train_loader_end) {
            torch::Tensor data, targets;
            auto batch = *train_loader_iterator;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() <<
                        std::endl;
            }
            ++train_loader_iterator;
        }
    }

    return 0;
}
