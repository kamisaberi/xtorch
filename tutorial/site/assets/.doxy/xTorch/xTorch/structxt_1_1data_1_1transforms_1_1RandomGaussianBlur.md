

# Struct xt::data::transforms::RandomGaussianBlur



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md) **>** [**RandomGaussianBlur**](structxt_1_1data_1_1transforms_1_1RandomGaussianBlur.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**RandomGaussianBlur**](#function-randomgaussianblur) (std::vector&lt; int &gt; sizes={3, 5}, double sigma\_min=0.1, double sigma\_max=2.0) <br> |
|  torch::Tensor | [**operator()**](#function-operator()) (const torch::Tensor & input\_tensor) <br> |




























## Public Functions Documentation




### function RandomGaussianBlur 

```C++
xt::data::transforms::RandomGaussianBlur::RandomGaussianBlur (
    std::vector< int > sizes={3, 5},
    double sigma_min=0.1,
    double sigma_max=2.0
) 
```




<hr>



### function operator() 

```C++
torch::Tensor xt::data::transforms::RandomGaussianBlur::operator() (
    const torch::Tensor & input_tensor
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/transforms/gaussian.h`

