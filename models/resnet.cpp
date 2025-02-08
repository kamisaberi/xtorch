#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#define DEBUG_MODE true

using namespace std;

class CIFAR10 : public torch::data::Dataset<CIFAR10> {
public:
    CIFAR10(const std::string &root) {
        // Load data from the specified root directory
        load_data(root);
    }

    // Override the get method to return a sample
    torch::data::Example<> get(size_t index) override {
        // Return the tensor image and its corresponding label
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    // Override the size method to return the number of samples
    torch::optional<size_t> size() const override {
        return data.size();
    }

private:
    std::vector<torch::Tensor> data; // Store image data as tensors
    std::vector<int64_t> labels;      // Store labels

    void load_data(const std::string &root) {
        const int num_files = 5;
        for (int i = 1; i <= num_files; ++i) {
            std::string file_path = root + "/data_batch_" + std::to_string(i) + ".bin";
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;
            }

            for (int j = 0; j < 10000; ++j) {
                uint8_t label;
                file.read(reinterpret_cast<char *>(&label), sizeof(label));
                labels.push_back(static_cast<int64_t>(label));

                std::vector<uint8_t> image(3072); // 32x32x3 = 3072
                file.read(reinterpret_cast<char *>(image.data()), image.size());

                // Reshape the image to 3x32x32 and convert to a Torch tensor
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                     torch::kByte).clone(); // Clone to ensure memory management
                tensor_image = tensor_image.permute({0, 2, 1}); // Permute to get the correct order (C, H, W)

                data.push_back(tensor_image); // Store the tensor in the data vector
            }

            file.close();
        }
    }
};

//class ResidualBlock(nn.Module):
//    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
//        super(ResidualBlock, self).__init__()
//        self.conv1 = nn.Sequential(
//                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
//                nn.BatchNorm2d(out_channels),
//                nn.ReLU())
//        self.conv2 = nn.Sequential(
//                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
//                nn.BatchNorm2d(out_channels))
//        self.downsample = downsample
//        self.relu = nn.ReLU()
//        self.out_channels = out_channels
//
//    def forward(self, x):
//        residual = x
//        out = self.conv1(x)
//        out = self.conv2(out)
//        if self.downsample:
//            residual = self.downsample(x)
//        out += residual
//                out = self.relu(out)
//        return out

struct ResidualBlock : torch::nn::Module {
    torch::nn::Sequential conv1 = nullptr, conv2 = nullptr, downsample = nullptr;
    int out_channels;
    torch::nn::ReLU relu= nullptr;
    torch::Tensor residual;

    ResidualBlock(int in_channels, int out_channels, int stride = 1, torch::nn::Sequential downsample = nullptr) {
        conv1 = torch::nn::Sequential();
        //                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
        torch::nn::Conv2d cnv1 = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1));
        conv1->push_back(cnv1);
        //                nn.BatchNorm2d(out_channels),
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(out_channels);
        conv1->push_back(batch1);
        //                nn.ReLU())
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        conv1->push_back(relu1);
        register_module("conv1", conv1);

        conv2 = torch::nn::Sequential();
        //                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        torch::nn::Conv2d cnv2 = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1));
        conv2->push_back(cnv2);

        //                nn.BatchNorm2d(out_channels),
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(out_channels);
        conv2->push_back(batch2);
        this->downsample = downsample;
        this->relu = torch::nn::ReLU();
        this->out_channels = out_channels;
    }

    torch::Tensor forward(torch::Tensor x) {
        residual = x;
        torch::Tensor out = conv1->forward(x);
        out = conv2->forward(out);
        if (downsample) {
            residual = downsample->forward(x);
        } else {}
        out += residual;
        out = relu(out);
        return out;
    }

};


struct Net : torch::nn::Module {
    int inplanes = 64;
    torch::nn::Sequential conv1 = nullptr;
    torch::nn::MaxPool2d maxpool= nullptr;
    torch::nn::AvgPool2d avgpool= nullptr;
    torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
    torch::nn::Linear fc = nullptr;

    Net(vector<int> layers, int num_classes = 10) {
        inplanes = 64;
        conv1 = torch::nn::Sequential();
//                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
        torch::nn::Conv2d cnv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3));
        conv1->push_back(cnv1);
//                nn.BatchNorm2d(64),
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(64);
        conv1->push_back(batch1);
        //                nn.ReLU())
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        conv1->push_back(relu1);
        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        register_module("layer0", layer0);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        register_module("layer1", layer1);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        register_module("layer2", layer2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);
        register_module("layer3", layer3);
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
    }

    torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1) {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes) {
            downsample = torch::nn::Sequential();
            //                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
            torch::nn::Conv2d convd = torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride).padding(0));
            downsample->push_back(convd);
            //                    nn.BatchNorm2d(planes),
            torch::nn::BatchNorm2d batchd = torch::nn::BatchNorm2d(planes);
            downsample->push_back(batchd);
        }
        torch::nn::Sequential layers = torch::nn::Sequential();
        ResidualBlock rb = ResidualBlock(inplanes, planes, stride, downsample);
        layers->push_back(rb);
        inplanes = planes;
        for (int i = 1; i < blocks; i++) {
            ResidualBlock rbt = ResidualBlock(inplanes, planes);
            layers->push_back(rbt);
        }
        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        cout << "1:" <<  x.sizes() << endl;
        x = conv1->forward(x);
        cout << "2:"<<  x.sizes() << endl;
        x = maxpool->forward(x);
        cout << "3:"<<  x.sizes() << endl;
        x = layer0->forward(x);
        cout << "4:"<<  x.sizes() << endl;
        x = layer1->forward(x);
        cout << "5:"<<  x.sizes() << endl;
        x = layer2->forward(x);
        cout << "6:"<<  x.sizes() << endl;
        x = layer3->forward(x);
        cout << "7:"<<  x.sizes() << endl;
        x = avgpool->forward(x);
        cout << "8:"<<  x.sizes() << endl;
        x = x.view({x.size(0),-1});
        cout << "9:"<<  x.sizes() << endl;
        x = fc->forward(x);
        cout << "10:"<<  x.sizes() << endl;
        return x;
    }
//class ResNet(nn.Module):
//    def __init__(self, block, layers, num_classes = 10):
//        super(ResNet, self).__init__()
//        self.inplanes = 64
//        self.conv1 = nn.Sequential(
//                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
//                nn.BatchNorm2d(64),
//                nn.ReLU())
//        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
//        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
//        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
//        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
//        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
//        self.avgpool = nn.AvgPool2d(7, stride=1)
//        self.fc = nn.Linear(512, num_classes)
//
//    def _make_layer(self, block, planes, blocks, stride=1):
//        downsample = None
//        if stride != 1 or self.inplanes != planes:
//            downsample = nn.Sequential(
//                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
//                    nn.BatchNorm2d(planes),
//            )
//        layers = []
//        layers.append(block(self.inplanes, planes, stride, downsample))
//        self.inplanes = planes
//        for i in range(1, blocks):
//            layers.append(block(self.inplanes, planes))
//
//        return nn.Sequential(*layers)
//
//    def forward(self, x):
//        x = self.conv1(x)
//        x = self.maxpool(x)
//        x = self.layer0(x)
//        x = self.layer1(x)
//        x = self.layer2(x)
//        x = self.layer3(x)
//        x = self.avgpool(x)
//        x = x.view(x.size(0), -1)
//        x = self.fc(x)
//        return x

};

void set_random() {
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}


// Function to resize a single tensor
torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
    return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
    ).squeeze(0);
}

//torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& size) {
//    // Check if the tensor is in the format [C, H, W]
//    cout << "inside: " <<  tensor.sizes() << endl;
//    TORCH_CHECK(tensor.size(0) == 3, "Input tensor must have 3 channels for RGB.");
//
//    auto out =  torch::nn::functional::interpolate(
//            tensor.unsqueeze(0), // Add a batch dimension
//            torch::nn::functional::InterpolateFuncOptions()
//                    .size(size)
//                    .mode(torch::kBilinear)
//                    .align_corners(false)
//    ).squeeze(0); // Remove the batch dimension
//
//    cout << "inside: " <<  out.sizes() << endl;
//    return  out;
//
//}

int main() {
    std::string dataset_path = "/home/kami/datasets/cifar-10-batches-bin/"; // Update with your path
    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, {224, 224});
                return example;
            }
    );

    CIFAR10 cifar10(dataset_path);

    // Create a data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cifar10.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
                    torch::data::transforms::Stack<>())),
            /*batch_size=*/64);


    Net model({ 3, 4, 6, 3},10);
    model.to(device);
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;


    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;
        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();
        while (train_loader_interator != train_loader_end) {
            torch::Tensor data, targets;
            auto batch = *train_loader_interator;

            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);

            torch::Tensor loss;

//            cout << output.sizes() << " " << targets.sizes() << endl;
//            cout << targets << endl;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();


            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>()
                          << std::endl;
            }
            ++train_loader_interator;

        }
    }




    // Print the size of the original and resized images
//    std::cout << "Original image size: " << train_loader[0].data.sizes() << std::endl;
//    std::cout << "Resized image size: " << transformed_dataset[0].data.sizes() << std::endl;

    return 0;
}
