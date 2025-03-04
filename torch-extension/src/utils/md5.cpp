#include "../../include/utils/md5.h"


std::string get_md5_checksum(const std::string &filename) {
    unsigned char mdValue[EVP_MAX_MD_SIZE];
    unsigned int mdLength;

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) {
        std::cerr << "Failed to create MD_CTX." << std::endl;
        return "";
    }

    if (EVP_DigestInit_ex(mdctx, EVP_md5(), nullptr) != 1) {
        ERR_print_errors_fp(stderr);
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    std::ifstream file(filename, std::ifstream::binary);
    if (!file) {
        std::cerr << "Could not open file: " << filename << std::endl;
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    char buffer[1024];
    while (file.read(buffer, sizeof(buffer))) {
        EVP_DigestUpdate(mdctx, buffer, file.gcount());
    }
    EVP_DigestUpdate(mdctx, buffer, file.gcount());

    if (EVP_DigestFinal_ex(mdctx, mdValue, &mdLength) != 1) {
        ERR_print_errors_fp(stderr);
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    EVP_MD_CTX_free(mdctx);

    std::ostringstream result;
    for (unsigned int i = 0; i < mdLength; ++i) {
        result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mdValue[i]);
    }
    return result.str();
}

//
//
//
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
