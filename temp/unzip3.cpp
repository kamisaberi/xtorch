#include <iostream>

#include "../includes/archiver.h"

using namespace  std;


int main(int argc, char* argv[]) {
    string gzFileName("/home/kami/Documents/b.tar.gz");
    string zipFileName("/home/kami/Documents/a.zip");
    const char *tarFile = "./test.tar";  // Path to the .tar file
//    extract_zip(argv[1]);
    extractZip(zipFileName);
    extractGzip(gzFileName , tarFile);
//    extractTar("test.tar", (const char* )"./");

    const char *extractDir = "./"; // Destination directory

    extractTar(tarFile);

    return 0;
}
