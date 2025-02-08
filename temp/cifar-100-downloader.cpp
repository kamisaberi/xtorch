#include <stdio.h>
#include <iostream>
#include <curl/curl.h>
using  namespace std;

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    // Write data to a file or buffer here if needed
    return size * nmemb; // Return the number of bytes processed
}

void ProgressCallback(double totalToDownload, double downloaded, double totalToUpload, double uploaded) {
    if (totalToDownload > 0) {
        int progress = static_cast<int>((downloaded / totalToDownload) * 100);
        std::cout << "\rDownload progress: " << progress << "%";
        std::cout.flush();
    }
}

int main(void)
{
    CURL *curl;
    FILE *fp;
    CURLcode res;
    string url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
    char outfilename[FILENAME_MAX] = "cifar-100-binary.tar.gz";
    curl = curl_easy_init();
    if (curl)
    {
        fp = fopen(outfilename,"wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
//        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);

        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);


        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
    }
    return 0;
}
//
// Created by kami on 1/29/25.
//
