#include <iostream>
#include <filesystem>
using namespace std;

int main(){

    std::filesystem::path base("/home/kami/datasets");
    std::filesystem::path folder("cifar-100-binary");
    std::filesystem::path file("train.bin");

    std::filesystem::path abs= base / folder / file;
    cout << abs << endl;
    cout << abs.string() << endl;




    return 0;
}