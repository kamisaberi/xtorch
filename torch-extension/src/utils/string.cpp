#include "../../include/utils/string.h"

namespace torch::ext::utils::string {
    vector<std::string> split(const std::string& str, const std::string& delim)
    {
        vector<std::string> tokens;
        size_t prev = 0, pos = 0;
        do
        {
            pos = str.find(delim, prev);
            if (pos == std::string::npos) pos = str.length();
            std::string token = str.substr(prev, pos-prev);
            if (!token.empty()) tokens.push_back(token);
            prev = pos + delim.length();
        }
        while (pos < str.length() && prev < str.length());
        return tokens;
    }
    std::string trim(std::string& str)
    {
        str.erase(str.find_last_not_of(' ')+1);         //suffixing spaces
        str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
        return str;
    }

}