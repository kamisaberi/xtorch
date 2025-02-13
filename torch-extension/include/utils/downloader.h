#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <curl/curl.h>
using namespace std;
namespace  fs = std::filesystem;

std::tuple<bool , std::string> download_data(std::string  &url, std::string outPath);
//inline std::tuple<bool , std::string> download_data(std::string  &url, std::string outPath) {
//    CURL *curl;
//    FILE *fp;
//    CURLcode res;
////                const char *url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
////    char outfilename[FILENAME_MAX] = "cifar-100-binary.tar.gz";
////    char outfilename[FILENAME_MAX] = "cifar-100-binary.tar.gz";
//    curl = curl_easy_init();
//    if (curl) {
//        string outFile=  (fs::path(outPath) /  fs::path(url).filename()).string();
//        fp = fopen(outFile.c_str(), "wb");
//        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
////        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
//        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
////        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, ProgressCallback);
//        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
//        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
//        res = curl_easy_perform(curl);
//        curl_easy_cleanup(curl);
//        fclose(fp);
//        return std::make_tuple(true,outFile) ;
//    }
//    return std::make_tuple(false, "");
//}
//
