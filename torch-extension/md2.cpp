#include "include/utils/md5.h"
//std::string get_md5_checksum(const std::string &filename) {
//    unsigned char mdValue[EVP_MAX_MD_SIZE];
//    unsigned int mdLength;
//
//    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
//    if (!mdctx) {
//        std::cerr << "Failed to create MD_CTX." << std::endl;
//        return "";
//    }
//
//    if (EVP_DigestInit_ex(mdctx, EVP_md5(), nullptr) != 1) {
//        ERR_print_errors_fp(stderr);
//        EVP_MD_CTX_free(mdctx);
//        return "";
//    }
//
//    std::ifstream file(filename, std::ifstream::binary);
//    if (!file) {
//        std::cerr << "Could not open file: " << filename << std::endl;
//        EVP_MD_CTX_free(mdctx);
//        return "";
//    }
//
//    char buffer[1024];
//    while (file.read(buffer, sizeof(buffer))) {
//        EVP_DigestUpdate(mdctx, buffer, file.gcount());
//    }
//    EVP_DigestUpdate(mdctx, buffer, file.gcount());
//
//    if (EVP_DigestFinal_ex(mdctx, mdValue, &mdLength) != 1) {
//        ERR_print_errors_fp(stderr);
//        EVP_MD_CTX_free(mdctx);
//        return "";
//    }
//
//    EVP_MD_CTX_free(mdctx);
//
//    std::ostringstream result;
//    for (unsigned int i = 0; i < mdLength; ++i) {
//        result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mdValue[i]);
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
