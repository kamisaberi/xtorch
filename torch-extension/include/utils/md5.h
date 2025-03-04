#pragma once

#include <iostream>
#include <fstream>
//#include <openssl/md5.h>
#include <iomanip>
#include <openssl/evp.h>
#include <openssl/err.h>


std::string get_md5_checksum(const std::string &filename) ;