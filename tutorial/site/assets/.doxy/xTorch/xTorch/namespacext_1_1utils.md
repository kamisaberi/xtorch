

# Namespace xt::utils



[**Namespace List**](namespaces.md) **>** [**xt**](namespacext.md) **>** [**utils**](namespacext_1_1utils.md)


















## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**fs**](namespacext_1_1utils_1_1fs.md) <br> |
| namespace | [**string**](namespacext_1_1utils_1_1string.md) <br> |
























## Public Functions

| Type | Name |
| ---: | :--- |
|  std::tuple&lt; bool, std::string &gt; | [**download**](#function-download) (std::string & url, std::string outPath) <br> |
|  std::tuple&lt; bool, std::string &gt; | [**download\_from\_gdrive**](#function-download_from_gdrive) (std::string gid, std::string outPath) <br> |
|  bool | [**extract**](#function-extract) (const std::string & inFile, const std::string & outFile="") <br> |
|  std::tuple&lt; bool, string &gt; | [**extractGzip**](#function-extractgzip) (const std::string & inFile, const std::string & outFile="") <br> |
|  bool | [**extractTar**](#function-extracttar) (const std::string & tarFile, const std::string & outPath="./") <br> |
|  bool | [**extractTgz**](#function-extracttgz) (const std::string & inFile, const std::string & outPath="./") <br> |
|  void | [**extractXZ**](#function-extractxz) (const std::string & inputFile, const std::string & outputFile) <br> |
|  bool | [**extractZip**](#function-extractzip) (const std::string & inFile, const std::string & outPath="./") <br> |
|  std::string | [**get\_md5\_checksum**](#function-get_md5_checksum) (const std::string & filename) <br> |
|  std::string | [**rebuild\_google\_drive\_link**](#function-rebuild_google_drive_link) (std::string gid) <br> |




























## Public Functions Documentation




### function download 

```C++
std::tuple< bool, std::string > xt::utils::download (
    std::string & url,
    std::string outPath
) 
```




<hr>



### function download\_from\_gdrive 

```C++
std::tuple< bool, std::string > xt::utils::download_from_gdrive (
    std::string gid,
    std::string outPath
) 
```




<hr>



### function extract 

```C++
bool xt::utils::extract (
    const std::string & inFile,
    const std::string & outFile=""
) 
```




<hr>



### function extractGzip 

```C++
std::tuple< bool, string > xt::utils::extractGzip (
    const std::string & inFile,
    const std::string & outFile=""
) 
```




<hr>



### function extractTar 

```C++
bool xt::utils::extractTar (
    const std::string & tarFile,
    const std::string & outPath="./"
) 
```




<hr>



### function extractTgz 

```C++
bool xt::utils::extractTgz (
    const std::string & inFile,
    const std::string & outPath="./"
) 
```




<hr>



### function extractXZ 

```C++
void xt::utils::extractXZ (
    const std::string & inputFile,
    const std::string & outputFile
) 
```




<hr>



### function extractZip 

```C++
bool xt::utils::extractZip (
    const std::string & inFile,
    const std::string & outPath="./"
) 
```




<hr>



### function get\_md5\_checksum 

```C++
std::string xt::utils::get_md5_checksum (
    const std::string & filename
) 
```




<hr>



### function rebuild\_google\_drive\_link 

```C++
std::string xt::utils::rebuild_google_drive_link (
    std::string gid
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/utils/downloader.h`

