

# Struct xt::data::transforms::GrayscaleToRGB



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md) **>** [**GrayscaleToRGB**](structxt_1_1data_1_1transforms_1_1GrayscaleToRGB.md)



_A functor to convert a grayscale tensor to an RGB tensor._ [More...](#detailed-description)

* `#include <grayscale.h>`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  torch::Tensor | [**operator()**](#function-operator()) (const torch::Tensor & tensor) <br>_Converts a grayscale tensor to an RGB tensor._  |




























## Detailed Description


This struct provides a callable object that transforms a grayscale tensor, typically with a single channel (e.g., [H, W] or [1, H, W]), into an RGB tensor with three channels (e.g., [3, H, W]). The conversion is performed by replicating the grayscale channel across the RGB dimensions, suitable for preprocessing grayscale images in machine learning workflows using LibTorch. 


    
## Public Functions Documentation




### function operator() 

_Converts a grayscale tensor to an RGB tensor._ 
```C++
torch::Tensor xt::data::transforms::GrayscaleToRGB::operator() (
    const torch::Tensor & tensor
) 
```





**Parameters:**


* `tensor` The input grayscale tensor, expected in format [H, W] or [1, H, W]. 



**Returns:**

A new tensor in RGB format [3, H, W], with the grayscale values replicated across channels.


This operator takes a grayscale tensor and produces an RGB tensor by duplicating the single channel’s values into three identical channels (red, green, blue). The input tensor must have a single channel, either as a 2D tensor [H, W] or a 3D tensor with one channel [1, H, W]. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/transforms/grayscale.h`

