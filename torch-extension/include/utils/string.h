#pragma once
#include <iostream>
//#include <string>
#include <sstream>
#include <vector>
using namespace std;
namespace xt::utils::string {
    vector<std::string> split(const std::string& str, const std::string& delim);
    std::string trim(std::string& str);

}


