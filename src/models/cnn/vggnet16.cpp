#include "../../../include/models/cnn/vggnet16.h"


VggNet16::VggNet16(int num_classes) {
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
    layer12 = torch::nn::Sequential();
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
    torch::nn::Linear linear20 = torch::nn::Linear(7 * 7 * 512, 4096);
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

torch::Tensor VggNet16::forward(torch::Tensor x) {
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
    x = x.view({x.size(0), -1});
    x = fc->forward(x);
    x = fc1->forward(x);
    x = fc2->forward(x);
    return x;
}


void set_random() {
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}

