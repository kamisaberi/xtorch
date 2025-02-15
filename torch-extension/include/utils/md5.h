#pragma once

#include <iostream>
#include <fstream>
//#include <openssl/md5.h>
#include <iomanip>
#include <openssl/evp.h>
#include <openssl/err.h>


std::string md5File(const std::string &filename) ;