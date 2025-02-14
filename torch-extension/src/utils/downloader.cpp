#include "../../include/utils/downloader.h"


using namespace std;
namespace  fs = std::filesystem;


size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    std::string header(buffer, size * nitems);

    // Look for the Content-Length header
    if (header.find("Content-Length:") != std::string::npos) {
        size_t pos = header.find(":") + 1;
        std::string length = header.substr(pos);
        // Trim whitespace
        length.erase(0, length.find_first_not_of(" \t\n\r"));
        length.erase(length.find_last_not_of(" \t\n\r") + 1);

        // Convert to size_t and store in userdata
        *(static_cast<size_t*>(userdata)) = std::stoul(length);
    }

    return nitems * size;
}


std::tuple<bool , std::string> download_data(std::string  &url, std::string outPath) {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    size_t file_size = 0;
//                const char *url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
//    char outfilename[FILENAME_MAX] = "cifar-100-binary.tar.gz";
//    char outfilename[FILENAME_MAX] = "cifar-100-binary.tar.gz";
    curl = curl_easy_init();
    if (curl) {
        string outFile=  (fs::path(outPath) /  fs::path(url).filename()).string();
        fp = fopen(outFile.c_str(), "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
//        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &file_size);
//        cout << "FILE_SIZE:" <<file_size << endl;

//        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
//        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
        return std::make_tuple(true,outFile) ;
    }
    return std::make_tuple(false, "");
}

