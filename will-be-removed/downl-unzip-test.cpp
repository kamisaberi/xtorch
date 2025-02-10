#include <stdio.h>
#include <iostream>
#include <curl/curl.h>
#include "../includes/downloader.h"
#include "../includes/archiver.h"
using  namespace std;


int main(void)
{

    string url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
//    string url = "test/ali/test.tar";
//    cout << std::filesystem::path(url).filename().string() << endl;
//    cout << std::filesystem::path(url).filename()<< endl;
//    cout << std::filesystem::path(url).parent_path()<< endl;
//    return  0;

    auto [result , path] =  download_data(url, "/home/kami/Documents/temp/d1/");
    if (result){
        extract(path , "");
    }

//    string path = "/home/kami/Documents/temp/cifar-100-binary.zip";
//    extract(path,"");
    return 0;
}
