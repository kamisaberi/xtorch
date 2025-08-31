# Computer Vision Datasets

xTorch provides an extensive collection of built-in dataset handlers for a wide variety of computer vision tasks, from image classification and object detection to semantic segmentation and beyond. This allows you to easily benchmark models on standard academic datasets without writing custom data loading code.

All computer vision datasets are located under the `xt::datasets` namespace and can be found within the `<xtorch/datasets/computer_vision/>` header directory.

## General Usage

The standard workflow for using any computer vision dataset involves defining a pipeline of image transformations, instantiating the desired dataset class, and then passing it to a data loader.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Define a pipeline of image transformations for data augmentation
    auto transforms = std::make_unique<xt::transforms::Compose>(
        std::make_shared<xt::transforms::image::RandomHorizontalFlip>(),
        std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}),
        std::make_shared<xt::transforms::general::Normalize>(
            std::vector<float>{0.5, 0.5, 0.5},
            std::vector<float>{0.5, 0.5, 0.5}
        )
    );

    // 2. Instantiate a dataset for CIFAR-10
    auto dataset = xt::datasets::CIFAR10(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(transforms)
    );

    std::cout << "CIFAR-10 dataset size: " << *dataset.size() << std::endl;

    // 3. Pass the dataset to a DataLoader
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 128, true, 4);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        auto images = batch.first;
        auto labels = batch.second;
        // ... training step ...
    }
}
```

!!! info "Standard Dataset Constructors"
Most dataset constructors follow a standard pattern:
`DatasetName(const std::string& root, DataMode mode, bool download, TransformPtr transforms)`
- `root`: The directory where the data is stored or will be downloaded.
- `mode`: `DataMode::TRAIN`, `DataMode::TEST`, or `DataMode::VALIDATION`.
- `download`: If `true`, the dataset will be downloaded if not found in the root directory.
- `transforms`: A `unique_ptr` to a transform pipeline to be applied to the data.

---

## Available Datasets by Task

### Image Classification

| Dataset Class | Description | Header File |
|---|---|---|
| `MNIST` | Grayscale handwritten digits (0-9). | `image_classification/mnist.h` |
| `FashionMNIST` | Grayscale images of 10 fashion categories. | `image_classification/fashion_mnist.h` |
| `KMNIST` | Kuzushiji-MNIST, a dataset of classical Japanese characters. | `image_classification/kmnist.h` |
| `EMNIST` | Extended MNIST, a larger set of handwritten letters and digits. | `image_classification/emnist.h` |
| `QMNIST` | A larger, cleaner version of the MNIST dataset. | `image_classification/qmnist.h` |
| `USPS` | A dataset of handwritten digits from the USPS. | `image_classification/usps.h` |
| `CIFAR10` | 32x32 color images in 10 classes. | `image_classification/cifar_10.h` |
| `CIFAR100` | 32x32 color images in 100 classes. | `image_classification/cifar_100.h` |
| `ImageNet` | The large-scale ImageNet (ILSVRC) dataset. | `image_classification/imagenet.h` |
| `CelebA` | Large-scale CelebFaces Attributes dataset. | `image_classification/celeba.h` |
| `STL10` | An image recognition dataset with 10 classes, with fewer labeled images than CIFAR-10. | `image_classification/stl.h` |
| `SVHN` | Street View House Numbers dataset. | `image_classification/svhn.h` |
| `Caltech101` | Images of objects belonging to 101 categories. | `image_classification/caltech101.h`|
| `Caltech256` | An improved version of Caltech101 with 256 categories. | `image_classification/caltech256.h`|
| `Food101` | A challenging dataset of 101 food categories. | `image_classification/food.h` |
| `Flowers102` | A dataset of 102 flower categories. | `image_classification/flowers.h` |
| `StanfordCars`| A dataset of 196 classes of cars. | `image_classification/stanford_cars.h` |
| `FGVCAircraft`| A fine-grained dataset of aircraft variants. | `image_classification/fgvc_aircraft.h` |
| `DTD` | Describable Textures Dataset for texture recognition. | `image_classification/dtd.h` |
| `EuroSAT` | A dataset of Sentinel-2 satellite images covering 10 land use classes. | `image_classification/euro_sat.h` |
| `GTSRB` | German Traffic Sign Recognition Benchmark. | `image_classification/gtsrb.h` |
| `PCAM` | PatchCamelyon, a medical imaging dataset for metastasis detection. | `image_classification/pcam.h` |
| `LFWPeople` | Labeled Faces in the Wild, a dataset for face recognition. | `image_classification/lfw_people.h` |

### Object Detection

| Dataset Class | Description | Header File |
|---|---|---|
| `COCODetection` | The popular COCO (Common Objects in Context) dataset for detection. | `object_detection/coco_detection.h`|
| `VOCDetection` | The PASCAL VOC dataset for object detection. | `object_detection/voc_detection.h` |
| `KITTI` | A popular dataset for autonomous driving research, including object detection. | `object_detection/kitti.h` |
| `OpenImages` | A large-scale dataset with millions of images and bounding boxes. | `object_detection/open_images.h`|
| `WIDERFace` | A face detection benchmark dataset. | `face_detection/wider_face.h` |

### Semantic & Instance Segmentation

| Dataset Class | Description | Header File |
|---|---|---|
| `VOCSegmentation`| The PASCAL VOC dataset for semantic segmentation. | `semantic_segmentation/voc_segmentation.h` |
| `Cityscapes` | A large-scale dataset focusing on semantic understanding of urban street scenes. | `semantic_segmentation/cityscapes.h` |
| `ADE20K` | A scene parsing benchmark for semantic segmentation and scene recognition. | `semantic_segmentation/ade20k.h` |
| `OxfordIIITPet` | A 37 category pet dataset with pixel-level segmentation masks. | `semantic_segmentation/oxfordIII_t_pet.h` |
| `LVIS` | A large vocabulary instance segmentation dataset. | `instance_segmentation/lvis.h` |

### Image Generation
| Dataset Class | Description | Header File |
|---|---|---|
| `FFHQ` | Flickr-Faces-HQ, a high-quality image dataset of human faces. | `image_generation/ffhq.h` |
| `CelebA` | The CelebA dataset, also commonly used for training GANs. | `image_classification/celeba.h` |

### Image Captioning
| Dataset Class | Description | Header File |
|---|---|---|
| `COCOCaptions` | The COCO dataset with its associated image captions. | `image_captioning/coco_captions.h` |
| `Flickr8k` | A dataset of 8,000 captioned images. | `image_classification/flickr_8k.h` |
| `Flickr30k` | A larger version of the Flickr dataset with 30,000 images. | `image_classification/flickr_30k.h` |

### Autonomous Driving & 3D Vision
| Dataset Class | Description | Header File |
|---|---|---|
| `WaymoOpenDataset`| A large and diverse dataset for autonomous driving research. | `autonomous_driving_perception/waymo_open_dataset.h` |
| `nuScenes` | A large-scale public dataset for autonomous driving. | `autonomous_driving_perception/nu_scenes.h` |
| `ModelNet40` | A dataset of 3D CAD models for point cloud analysis. | `3d_point_cloud_analysis/model_net40.h` |
| `ShapeNet` | A large repository of 3D shapes. | `3d_shape_generation/shapenet.h` |

### Optical Flow
| Dataset Class | Description | Header File |
|---|---|---|
| `FlyingChairs` | A synthetic dataset for training optical flow networks. | `optical_flow_estimation/flying_chairs.h`|
| `Sintel` | A popular benchmark for optical flow, with realistic rendering. | `optical_flow_estimation/sintel.h` |
