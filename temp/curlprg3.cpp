#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    ofstream* file = static_cast<ofstream*>(stream);
    size_t written = size * nmemb;
    file->write(static_cast<char*>(ptr), written);
    return written;
}

class PipStyleBar {
    int width;
    time_point<steady_clock> start;

public:
    PipStyleBar(int w = 50) : width(w), start(steady_clock::now()) {}

    void update(curl_off_t total, curl_off_t now) {
        if (total == 0) return;
        float ratio = static_cast<float>(now) / total;
        int filled = static_cast<int>(ratio * width);

        auto now_time = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now_time - start).count();
        float speed = elapsed > 0 ? now / elapsed / 1024 : 0;
        float eta = speed > 0 ? (total - now) / 1024 / speed : 0;

        // Red/pink color
        const string color = "\033[38;5;197m"; // Bright pink-red
        const string reset = "\033[0m";

        cout << "\r" << color << "[";
        for (int i = 0; i < width; ++i)
            cout << (i < filled ? "━" : " ");
        cout << "] " << setw(5) << fixed << setprecision(1) << (ratio * 100) << "% ";
        cout << setw(6) << setprecision(0) << speed << " KB/s ETA: " << setw(3) << eta << "s";
        cout << reset << flush;
    }
};

PipStyleBar bar;

int progress_callback(void*, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
    bar.update(total, now);
    return 0;
}

int main() {
    const string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"; // Example large file
    const string out = "cifar-10-binary.tar.gz";

    CURL* curl;
    CURLcode res;
    ofstream file(out, ios::binary);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl && file.is_open()) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);

        // Disable SSL check for demo
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        cout << "Downloading: " << url << endl;

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            cerr << "\n❌ Error: " << curl_easy_strerror(res) << endl;
        else
            cout << "\n✅ Done. Saved to " << out << endl;

        curl_easy_cleanup(curl);
    } else {
        cerr << "❌ Failed to start." << endl;
    }

    curl_global_cleanup();
    file.close();
    return 0;
}
