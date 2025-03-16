#include <iostream>
#include <fstream>
#include <openssl/md5.h>
#include <iomanip>
#include "include/utils/md5.h"
//std::string get_md5_checksum(const std::string &filename) {
//    unsigned char c[MD5_DIGEST_LENGTH];
//    MD5_CTX mdContext;
//    MD5_Init(&mdContext);
//
//    std::ifstream file(filename, std::ifstream::binary);
//    if (!file) {
//        std::cerr << "Could not open file: " << filename << std::endl;
//        return "";
//    }
//
//    char buffer[1024];
//    while (file.read(buffer, sizeof(buffer))) {
//        MD5_Update(&mdContext, buffer, file.gcount());
//    }
//    MD5_Update(&mdContext, buffer, file.gcount());
//    MD5_Final(c, &mdContext);
//
//    std::ostringstream result;
//    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
//        result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c[i]);
//    }
//    return result.str();
//}

int main() {
    std::string filename = "/home/kami/Documents/temp/cifar-100-binary.tar.gz"; // Change this to your file path
    std::string md5Hash = torch::ext::utils::get_md5_checksum(filename);

    if (!md5Hash.empty()) {
        std::cout << "MD5 Hash: " << md5Hash << std::endl;
    }

    return 0;
}
