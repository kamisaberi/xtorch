

# Struct xt::data::transforms::Resize



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md) **>** [**Resize**](structxt_1_1data_1_1transforms_1_1Resize.md)



_A functor to resize a tensor image to a specified size._ [More...](#detailed-description)

* `#include <resize.h>`





































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Resize**](#function-resize) (std::vector&lt; int64\_t &gt; size) <br>_Constructs a_ [_**Resize**_](structxt_1_1data_1_1transforms_1_1Resize.md) _object with the target size._ |
|  torch::Tensor | [**operator()**](#function-operator()) (torch::Tensor img) <br>_Resizes the input tensor image to the target size._  |




























## Detailed Description


This struct provides a callable object that resizes a `torch::Tensor` representing an image to a target size specified as a vector of 64-bit integers. It uses the call operator to perform the resizing operation, making it suitable for use in functional pipelines or transformations. 


    
## Public Functions Documentation




### function Resize 

_Constructs a_ [_**Resize**_](structxt_1_1data_1_1transforms_1_1Resize.md) _object with the target size._
```C++
xt::data::transforms::Resize::Resize (
    std::vector< int64_t > size
) 
```





**Parameters:**


* `size` A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}). 




        

<hr>



### function operator() 

_Resizes the input tensor image to the target size._ 
```C++
torch::Tensor xt::data::transforms::Resize::operator() (
    torch::Tensor img
) 
```





**Parameters:**


* `img` The input tensor image to be resized. 



**Returns:**

A new tensor with the resized dimensions. 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/transforms/resize.h`

