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
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32}, torch::kByte).clone(); // Clone to ensure memory management
                tensor_image = tensor_image.permute({0, 2, 1}); // Permute to get the correct order (C, H, W)

                data.push_back(tensor_image); // Store the tensor in the data vector
            }

            file.close();
        }
    }
};



//class VGG16(nn.Module):
//def __init__(self, num_classes=10):
//super(VGG16, self).__init__()
//self.layer1 = nn.Sequential(
//        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(64),
//        nn.ReLU())
//self.layer2 = nn.Sequential(
//        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(64),
//        nn.ReLU(),
//        nn.MaxPool2d(kernel_size = 2, stride = 2))
//self.layer3 = nn.Sequential(
//        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(128),
//        nn.ReLU())
//self.layer4 = nn.Sequential(
//        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(128),
//        nn.ReLU(),
//        nn.MaxPool2d(kernel_size = 2, stride = 2))
//self.layer5 = nn.Sequential(
//        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(256),
//        nn.ReLU())
//self.layer6 = nn.Sequential(
//        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(256),
//        nn.ReLU())
//self.layer7 = nn.Sequential(
//        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(256),
//        nn.ReLU(),
//        nn.MaxPool2d(kernel_size = 2, stride = 2))
//self.layer8 = nn.Sequential(
//        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU())
//self.layer9 = nn.Sequential(
//        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU())
//self.layer10 = nn.Sequential(
//        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU(),
//        nn.MaxPool2d(kernel_size = 2, stride = 2))
//self.layer11 = nn.Sequential(
//        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU())
//self.layer12 = nn.Sequential(
//        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU())
//self.layer13 = nn.Sequential(
//        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
//        nn.BatchNorm2d(512),
//        nn.ReLU(),
//        nn.MaxPool2d(kernel_size = 2, stride = 2))
//self.fc = nn.Sequential(
//        nn.Dropout(0.5),
//        nn.Linear(7*7*512, 4096),
//        nn.ReLU())
//self.fc1 = nn.Sequential(
//        nn.Dropout(0.5),
//        nn.Linear(4096, 4096),
//        nn.ReLU())
//self.fc2= nn.Sequential(
//        nn.Linear(4096, num_classes))
//
//def forward(self, x):
//out = self.layer1(x)
//out = self.layer2(out)
//out = self.layer3(out)
//out = self.layer4(out)
//out = self.layer5(out)
//out = self.layer6(out)
//out = self.layer7(out)
//out = self.layer8(out)
//out = self.layer9(out)
//out = self.layer10(out)
//out = self.layer11(out)
//out = self.layer12(out)
//out = self.layer13(out)
//out = out.reshape(out.size(0), -1)
//out = self.fc(out)
//out = self.fc1(out)
//out = self.fc2(out)
//return out
//



struct Net : torch::nn::Module {
    torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr , layer4 = nullptr , layer5 = nullptr ;
    torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr , layer9 = nullptr , layer10 = nullptr ;
    torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr  ;
    torch::nn::Sequential fc = nullptr , fc1 = nullptr , fc2 = nullptr;

    Net(int num_classes) {
        //TODO layer1 DONE
        layer1 = torch::nn::Sequential();
        //        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1));
        layer1->push_back(conv1);
        //        nn.BatchNorm2d(64),
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(64);
        layer1->push_back(batch1);
        //        nn.ReLU())
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        layer1->push_back(relu1);

        register_module("layer1", layer1);

        //TODO layer2 DONE
        layer2 = torch::nn::Sequential();
        //        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1));
        layer2->push_back(conv2);
        //        nn.BatchNorm2d(64),
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(64);
        layer2->push_back(batch2);
        //        nn.ReLU(),
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer2->push_back(pool2);
        register_module("layer2", layer2);

        //TODO layer3 DONE
        layer3 = torch::nn::Sequential();
        //        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1));
        layer3->push_back(conv3);
        //        nn.BatchNorm2d(128),
        torch::nn::BatchNorm2d batch3 = torch::nn::BatchNorm2d(128);
        layer3->push_back(batch3);
        //        nn.ReLU())
        torch::nn::ReLU relu3 = torch::nn::ReLU();
        layer3->push_back(relu3);
        register_module("layer3", layer3);

        //TODO layer4 DONE
        layer4 = torch::nn::Sequential();
        //        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1));
        layer4->push_back(conv4);
        //        nn.BatchNorm2d(128),
        torch::nn::BatchNorm2d batch4 = torch::nn::BatchNorm2d(128);
        layer4->push_back(batch4);
        //        nn.ReLU(),
        torch::nn::ReLU relu4 = torch::nn::ReLU();
        layer4->push_back(relu4);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer4->push_back(pool4);
        register_module("layer4", layer4);

        //TODO layer5 DONE
        layer5 = torch::nn::Sequential();
        //        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1));
        layer5->push_back(conv5);
        //        nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch5 = torch::nn::BatchNorm2d(256);
        layer5->push_back(batch5);
        //        nn.ReLU())
        torch::nn::ReLU relu5 = torch::nn::ReLU();
        layer5->push_back(relu5);
        register_module("layer5", layer5);


        //TODO layer6 DONE
        layer6 = torch::nn::Sequential();
        //        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1));
        layer6->push_back(conv6);
        //        nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch6 = torch::nn::BatchNorm2d(256);
        layer6->push_back(batch6);
        //        nn.ReLU())
        torch::nn::ReLU relu6 = torch::nn::ReLU();
        layer6->push_back(relu6);
        register_module("layer6", layer6);


        //TODO layer7 DONE
        layer7 = torch::nn::Sequential();
        //        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1));
        layer7->push_back(conv7);
        //        nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch7 = torch::nn::BatchNorm2d(256);
        layer7->push_back(batch7);
        //        nn.ReLU())
        torch::nn::ReLU relu7 = torch::nn::ReLU();
        layer7->push_back(relu7);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool7 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer7->push_back(pool7);
        register_module("layer7", layer7);



        //TODO layer8 DONE
        layer8 = torch::nn::Sequential();
        //        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv8 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1));
        layer8->push_back(conv8);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch8 = torch::nn::BatchNorm2d(512);
        layer8->push_back(batch8);
        //        nn.ReLU())
        torch::nn::ReLU relu8 = torch::nn::ReLU();
        layer8->push_back(relu8);
        register_module("layer8", layer8);


        //TODO layer9 DONE
        layer9 = torch::nn::Sequential();
        //        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv9 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1));
        layer9->push_back(conv9);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch9 = torch::nn::BatchNorm2d(512);
        layer9->push_back(batch9);
        //        nn.ReLU())
        torch::nn::ReLU relu9 = torch::nn::ReLU();
        layer9->push_back(relu9);
        register_module("layer9", layer9);


        //TODO layer10 DONE
        //self.layer10 = nn.Sequential(
        layer10 = torch::nn::Sequential();
        //        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv10 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1));
        layer10->push_back(conv10);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch10 = torch::nn::BatchNorm2d(512);
        layer10->push_back(batch10);
        //        nn.ReLU())
        torch::nn::ReLU relu10 = torch::nn::ReLU();
        layer10->push_back(relu10);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool10 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer10->push_back(pool10);
        register_module("layer10", layer10);



        //TODO layer11 DONE
        //self.layer11 = nn.Sequential(
        layer11 = torch::nn::Sequential();
        //        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv11 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1));
        layer11->push_back(conv11);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch11 = torch::nn::BatchNorm2d(512);
        layer11->push_back(batch11);
        //        nn.ReLU())
        torch::nn::ReLU relu11 = torch::nn::ReLU();
        layer11->push_back(relu11);
        register_module("layer11", layer11);


        //TODO layer12 DONE
        //self.layer12 = nn.Sequential(
        layer12= torch::nn::Sequential();
        //        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv12 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1));
        layer12->push_back(conv12);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch12 = torch::nn::BatchNorm2d(512);
        layer12->push_back(batch12);
        //        nn.ReLU())
        torch::nn::ReLU relu12 = torch::nn::ReLU();
        layer12->push_back(relu12);
        register_module("layer12", layer12);


        //TODO layer13
        //self.layer13 = nn.Sequential(
        layer13 = torch::nn::Sequential();
        //        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv13 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1));
        layer13->push_back(conv13);
        //        nn.BatchNorm2d(512),
        torch::nn::BatchNorm2d batch13 = torch::nn::BatchNorm2d(512);
        layer13->push_back(batch13);
        //        nn.ReLU())
        torch::nn::ReLU relu13 = torch::nn::ReLU();
        layer13->push_back(relu13);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool13 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer13->push_back(pool13);
        register_module("layer13", layer13);



        //TODO fc DONE
        fc = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop20 = torch::nn::Dropout(0.5);
        fc->push_back(drop20);
        //             nn.Linear(7*7*512, 4096),
        torch::nn::Linear linear20 = torch::nn::Linear(7*7*512, 4096);
        fc->push_back(linear20);
        //             nn.ReLU())
        torch::nn::ReLU relu20 = torch::nn::ReLU();
        fc->push_back(relu20);
        register_module("fc", fc);


        //TODO fc1 DONE
        fc1 = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop21 = torch::nn::Dropout(0.5);
        fc1->push_back(drop21);
        //             nn.Linear(4096, 4096),
        torch::nn::Linear linear21 = torch::nn::Linear(4096, 4096);
        fc1->push_back(linear21);
        //             nn.ReLU())
        torch::nn::ReLU relu21 = torch::nn::ReLU();
        fc1->push_back(relu21);
        register_module("fc1", fc1);

        //TODO fc2 DONE
        fc2 = torch::nn::Sequential();
        //        nn.Linear(4096, num_classes))
        torch::nn::Linear linear22 = torch::nn::Linear(4096, num_classes);
        fc2->push_back(linear22);
        register_module("fc2", fc2);


    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        x = x.view({x.size(0),-1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return  x;
    }

};



void set_random()
{
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}



// Function to resize a single tensor
torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& size) {
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
                example.data = resize_tensor(example.data, {227, 227});
                return example;
            }
    );

    CIFAR10 cifar10(dataset_path);

    // Create a data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cifar10.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
                    torch::data::transforms::Stack<>())),
            /*batch_size=*/64);


    Net model(10);
    model.to(device);
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;



    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;
        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();
        while(train_loader_interator != train_loader_end) {
            torch::Tensor  data,targets;
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
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
            }
            ++train_loader_interator;

        }
    }




    // Print the size of the original and resized images
//    std::cout << "Original image size: " << train_loader[0].data.sizes() << std::endl;
//    std::cout << "Resized image size: " << transformed_dataset[0].data.sizes() << std::endl;

    return 0;
}

