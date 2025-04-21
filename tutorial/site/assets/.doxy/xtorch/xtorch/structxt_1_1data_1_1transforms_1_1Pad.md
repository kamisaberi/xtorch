

# Struct xt::data::transforms::Pad



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md) **>** [**Pad**](structxt_1_1data_1_1transforms_1_1Pad.md)



_A functor to pad a tensor with a specified padding configuration._ [More...](#detailed-description)

* `#include <pad.h>`





































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Pad**](#function-pad) (std::vector&lt; int64\_t &gt; padding) <br>_Constructs a_ [_**Pad**_](structxt_1_1data_1_1transforms_1_1Pad.md) _object with the specified padding sizes._ |
|  torch::Tensor | [**operator()**](#function-operator()) (torch::Tensor input) <br>_Applies padding to the input tensor._  |




























## Detailed Description


This struct provides a callable object that applies padding to a `torch::Tensor` based on a vector of padding sizes. It is designed to extend the dimensions of a tensor (e.g., images in machine learning workflows) by adding values (typically zeros) around its boundaries, using the padding amounts specified during construction. The padding is applied using LibTorch's functional padding utilities. 


    
## Public Functions Documentation




### function Pad 

_Constructs a_ [_**Pad**_](structxt_1_1data_1_1transforms_1_1Pad.md) _object with the specified padding sizes._
```C++
xt::data::transforms::Pad::Pad (
    std::vector< int64_t > padding
) 
```





**Parameters:**


* `padding` A vector of 64-bit integers defining the padding amounts, in pairs (e.g., {left, right, top, bottom}).

Initializes the [**Pad**](structxt_1_1data_1_1transforms_1_1Pad.md) object with a vector specifying the padding to be applied to the tensor’s dimensions. The vector must contain an even number of elements, where each pair corresponds to the left and right padding for a dimension, applied to the tensor’s last dimensions in reverse order (e.g., width, then height for a 2D tensor). 


        

<hr>



### function operator() 

_Applies padding to the input tensor._ 
```C++
torch::Tensor xt::data::transforms::Pad::operator() (
    torch::Tensor input
) 
```





**Parameters:**


* `input` The input tensor to be padded, typically in format [N, C, H, W] or [H, W]. 



**Returns:**

A new tensor with padded dimensions according to the stored padding configuration.


This operator pads the input tensor using the padding sizes provided at construction, typically with zeros using constant mode padding. For a 4D tensor [N, C, H, W] and padding {p\_left, p\_right, p\_top, p\_bottom}, it pads width (W) and height (H), resulting in [N, C, H + p\_top + p\_bottom, W + p\_left + p\_right]. The padding is applied to the last dimensions corresponding to the number of pairs in the padding vector. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/transforms/pad.h`

