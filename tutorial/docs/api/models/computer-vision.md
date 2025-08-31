# Computer Vision Models

xTorch provides a rich and diverse model zoo for a wide array of computer vision tasks. These pre-built models allow you to quickly apply powerful, state-of-the-art architectures to your data.

All computer vision models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/computer_vision/>` directory.

## General Usage

Using a pre-built computer vision model is straightforward. You instantiate the model, typically providing task-specific parameters like the number of output classes, and then it's ready for training or inference.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Example: Instantiate a VGG16 model for a 10-class classification problem
    xt::models::VGGNet model(
        xt::models::VGGNetImpl::VGGType::VGG16,
        /*num_classes=*/10
    );

    model.to(device);
    model.train(); // Set to training mode

    // Create a dummy input batch (Batch=4, Channels=3, Height=224, Width=224)
    auto input_tensor = torch::randn({4, 3, 224, 224}).to(device);

    // Perform a forward pass
    auto output = model.forward(input_tensor);

    std::cout << "VGG16 Model Instantiated." << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl; // Should be
}
```

!!! info "Model Variants"
Many model families like `ResNet`, `VGGNet`, and `EfficientNet` have multiple variants (e.g., `ResNet18` vs. `ResNet50`). These are typically selected via an `enum` or by passing configuration arguments to the constructor. Please refer to the specific model's header file for all available options.

---

## Available Models by Task

### Image Classification

These models are designed to take an image as input and output a probability distribution over a set of classes.

| Model Family | Description | Header File |
|---|---|---|
| `LeNet5` | The classic LeNet-5 architecture, foundational for CNNs. | `image_classification/lenet5.h` |
| `AlexNet` | The breakthrough deep CNN from the 2012 ImageNet competition. | `image_classification/alexnet.h` |
| `VGGNet` | A simple and effective architecture with very small (3x3) convolution filters. | `image_classification/vggnet.h` |
| `ResNet` | Residual Networks, which introduced skip connections to enable much deeper models. | `image_classification/resnet.h` |
| `ResNeXt` | An evolution of ResNet that uses grouped convolutions. | `image_classification/resnext.h` |
| `WideResNet` | A ResNet variant that is wider (more channels) but shallower. | `image_classification/wide_resnet.h`|
| `GoogLeNet` | A deep CNN that introduced the "Inception" module. | `image_classification/google_net.h`|
| `Inception` | Later versions of the Inception architecture (e.g., InceptionV3). | `image_classification/inception.h`|
| `InceptionResNet`| A hybrid architecture combining Inception modules with residual connections. | `image_classification/inception_resnet.h` |
| `DenseNet` | Densely Connected Convolutional Networks, where each layer is connected to every other layer. | `image_classification/dense_net.h` |
| `MobileNet` | A family of efficient models for mobile and embedded vision applications. | `image_classification/mobilenet.h`|
| `EfficientNet` | A family of models that scales depth, width, and resolution in a principled way. | `image_classification/efficient_net.h` |
| `Xception` | An architecture based on depthwise separable convolutions. | `image_classification/xception.h` |
| `SENet` | Squeeze-and-Excitation Networks that adaptively recalibrate channel-wise feature responses. | `image_classification/se_net.h` |
| `CBAM` | Convolutional Block Attention Module. | `image_classification/cbam.h` |
| `NetworkInNetwork`| A model that uses micro neural networks in place of linear filters. | `image_classification/network_in_network.h`|
| `PyramidalNet` | A variant of ResNet that gradually increases feature map dimensions. | `image_classification/pyramidal_net.h`|
| `HighwayNetwork`| A deep network with learnable gating mechanisms. | `image_classification/highway_network.h`|
| `AmoebaNet` | An architecture discovered through evolutionary neural architecture search. | `image_classification/amoeba_net.h` |
| `ZefNet` | A visualization-driven model, an early winner of the ImageNet competition. | `image_classification/zefnet.h` |

### Object Detection

These models identify and locate multiple objects within an image by outputting bounding boxes and class labels.

| Model Family | Description | Header File |
|---|---|---|
| `RCNN` | Region-based CNN, the original groundbreaking model for this task. | `object_detection/rcnn.h` |
| `FastRCNN` | An improved version of R-CNN that is faster to train and test. | `object_detection/fast_rcnn.h` |
| `FasterRCNN` | Introduces a Region Proposal Network (RPN) for end-to-end training. | `object_detection/faster_rcnn.h` |
| `MaskRCNN` | An extension of Faster R-CNN that also adds a branch for predicting segmentation masks. | `object_detection/mask_rcnn.h` |
| `SSD` | Single Shot MultiBox Detector, a one-stage detector that is very fast. | `object_detection/ssd.h` |
| `RetinaNet` | A one-stage detector that introduced the Focal Loss to address class imbalance. | `object_detection/retina_net.h`|
| `YOLO` | You Only Look Once, a family of extremely fast one-stage detectors. | `object_detection/yolo.h` |
| `YOLOX` | An anchor-free version of YOLO. | `object_detection/yolox.h` |
| `DETR` | Detection Transformer, which frames object detection as a set prediction problem. | `object_detection/detr.h` |
| `EfficientDet` | A family of scalable and efficient object detectors. | `object_detection/efficient_det.h`|

### Image Segmentation

These models classify each pixel in an image to create a segmentation map.

| Model Family | Description | Header File |
|---|---|---|
| `FCN` | Fully Convolutional Network, a foundational model for semantic segmentation. | `image_segmentation/fcn.h` |
| `UNet` | An architecture with a U-shaped encoder-decoder structure, popular for biomedical imaging. | `image_segmentation/unet.h` |
| `SegNet` | A deep encoder-decoder architecture for semantic pixel-wise segmentation. | `image_segmentation/segnet.h` |
| `DeepLab` | A family of models (e.g., DeepLabV3+) using atrous convolutions for segmentation. | `image_segmentation/deep_lab.h` |
| `HRNet` | High-Resolution Network, which maintains high-resolution representations through the network. | `image_segmentation/hrnet.h` |
| `PANet` | Path Aggregation Network, which enhances feature fusion. | `image_segmentation/panet.h` |

### Vision Transformers

These models apply the Transformer architecture, originally designed for NLP, to computer vision tasks.

| Model Family | Description | Header File |
|---|---|---|
| `ViT` | Vision Transformer, the original model that applies a pure Transformer to image patches. | `vision_transformers/vit.h` |
| `DeiT` | Data-efficient Image Transformer, which uses knowledge distillation. | `vision_transformers/deit.h` |
| `SwinTransformer`| A hierarchical Vision Transformer using shifted windows. | `vision_transformers/swin_transformer.h`|
| `PVT` | Pyramid Vision Transformer, which introduces a pyramid structure to ViT. | `vision_transformers/pvt.h` |
| `T2TViT` | Token-to-Token Vision Transformer. | `vision_transformers/t2t_vit.h` |
| `MViT` | Multiscale Vision Transformer. | `vision_transformers/mvit.h` |
| `BEiT` | Bidirectional Encoder representation from Image Transformers (BERT pre-training for vision). | `vision_transformers/beit.h` |
| `CLIPViT` | The Vision Transformer backbone used in the CLIP model. | `vision_transformers/clip_vit.h` |
