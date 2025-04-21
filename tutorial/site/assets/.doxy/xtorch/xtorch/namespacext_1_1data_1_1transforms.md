

# Namespace xt::data::transforms



[**Namespace List**](namespaces.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md)




















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**CenterCrop**](structxt_1_1data_1_1transforms_1_1CenterCrop.md) <br> |
| struct | [**ColorJitter**](structxt_1_1data_1_1transforms_1_1ColorJitter.md) <br> |
| class | [**Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) <br>_A class to compose multiple tensor transformations into a single callable pipeline._  |
| struct | [**Cutout**](structxt_1_1data_1_1transforms_1_1Cutout.md) <br> |
| struct | [**GaussianBlur**](structxt_1_1data_1_1transforms_1_1GaussianBlur.md) <br> |
| struct | [**GaussianBlurOpenCV**](structxt_1_1data_1_1transforms_1_1GaussianBlurOpenCV.md) <br> |
| struct | [**GaussianNoise**](structxt_1_1data_1_1transforms_1_1GaussianNoise.md) <br> |
| struct | [**Grayscale**](structxt_1_1data_1_1transforms_1_1Grayscale.md) <br> |
| struct | [**GrayscaleToRGB**](structxt_1_1data_1_1transforms_1_1GrayscaleToRGB.md) <br>_A functor to convert a grayscale tensor to an RGB tensor._  |
| struct | [**HorizontalFlip**](structxt_1_1data_1_1transforms_1_1HorizontalFlip.md) <br> |
| struct | [**Lambda**](structxt_1_1data_1_1transforms_1_1Lambda.md) <br> |
| struct | [**Normalize**](structxt_1_1data_1_1transforms_1_1Normalize.md) <br> |
| struct | [**Pad**](structxt_1_1data_1_1transforms_1_1Pad.md) <br>_A functor to pad a tensor with a specified padding configuration._  |
| struct | [**RandomCrop**](structxt_1_1data_1_1transforms_1_1RandomCrop.md) <br> |
| struct | [**RandomCrop2**](structxt_1_1data_1_1transforms_1_1RandomCrop2.md) <br> |
| struct | [**RandomFlip**](structxt_1_1data_1_1transforms_1_1RandomFlip.md) <br> |
| struct | [**RandomGaussianBlur**](structxt_1_1data_1_1transforms_1_1RandomGaussianBlur.md) <br> |
| struct | [**Resize**](structxt_1_1data_1_1transforms_1_1Resize.md) <br>_A functor to resize a tensor image to a specified size._  |
| struct | [**Rotation**](structxt_1_1data_1_1transforms_1_1Rotation.md) <br> |
| struct | [**ToGray**](structxt_1_1data_1_1transforms_1_1ToGray.md) <br> |
| struct | [**ToTensor**](structxt_1_1data_1_1transforms_1_1ToTensor.md) <br> |
| struct | [**VerticalFlip**](structxt_1_1data_1_1transforms_1_1VerticalFlip.md) <br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  std::function&lt; torch::Tensor(torch::Tensor input)&gt; | [**create\_resize\_transform**](#function-create_resize_transform) (std::vector&lt; int64\_t &gt; size) <br> |
|  torch::data::transforms::Lambda&lt; torch::data::Example&lt;&gt; &gt; | [**normalize**](#function-normalize) (double mean, double stddev) <br> |
|  torch::data::transforms::Lambda&lt; torch::data::Example&lt;&gt; &gt; | [**resize**](#function-resize) (std::vector&lt; int64\_t &gt; size) <br> |
|  torch::Tensor | [**resize\_tensor**](#function-resize_tensor) (const torch::Tensor & tensor, const std::vector&lt; int64\_t &gt; & size) <br> |




























## Public Functions Documentation




### function create\_resize\_transform 

```C++
std::function< torch::Tensor(torch::Tensor input)> xt::data::transforms::create_resize_transform (
    std::vector< int64_t > size
) 
```




<hr>



### function normalize 

```C++
torch::data::transforms::Lambda< torch::data::Example<> > xt::data::transforms::normalize (
    double mean,
    double stddev
) 
```




<hr>



### function resize 

```C++
torch::data::transforms::Lambda< torch::data::Example<> > xt::data::transforms::resize (
    std::vector< int64_t > size
) 
```




<hr>



### function resize\_tensor 

```C++
torch::Tensor xt::data::transforms::resize_tensor (
    const torch::Tensor & tensor,
    const std::vector< int64_t > & size
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/definitions/transforms.h`

