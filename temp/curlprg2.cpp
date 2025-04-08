#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    ofstream* file = static_cast<ofstream*>(stream);
    size_t written = size * nmemb;
    file->write(static_cast<char*>(ptr), written);
    return written;
}

class ProgressBar {
    int bar_width;
    time_point<steady_clock> start_time;

public:
    ProgressBar(int width = 40) : bar_width(width), start_time(steady_clock::now()) {}

    void update(curl_off_t total, curl_off_t now) {
        if (total == 0) return;

        float progress = static_cast<float>(now) / total;
        int pos = static_cast<int>(bar_width * progress);

        auto now_time = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now_time - start_time).count();
        float speed = elapsed > 0 ? now / elapsed / 1024 : 0; // KB/s
        float eta = speed > 0 ? (total - now) / 1024 / speed : 0;

        cout << "\r[";

        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) cout << "█";
            else if (i == pos) cout << ">";
            else cout << " ";
        }

        cout << "] "
             << fixed << setprecision(1)
             << (progress * 100.0) << "% "
             << setw(6) << setprecision(0) << speed << " KB/s "
             << "ETA: " << setw(3) << setprecision(0) << eta << "s    ";

        cout.flush();
    }
};

// Global so CURL can access it
ProgressBar bar;

// Progress callback wrapper
int progress_callback(void*, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
    bar.update(total, now);
    return 0;
}

int main() {
    const string url = "https://speed.hetzner.de/100MB.bin";
    const string output_file = "downloaded_file.bin";

    CURL* curl;
    CURLcode res;
    ofstream file(output_file, ios::binary);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl && file.is_open()) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);

        // SSL skip for now (only for testing)
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        // Fancy progress bar
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);

        cout << "Downloading: " << url << endl;

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            cerr << "\nDownload failed: " << curl_easy_strerror(res) << endl;
        else
            cout << "\n✅ Download complete: " << output_file << endl;

        curl_easy_cleanup(curl);
    } else {
        cerr << "❌ Failed to initialize CURL or open file." << endl;
    }

    curl_global_cleanup();
    file.close();
    return 0;
}
