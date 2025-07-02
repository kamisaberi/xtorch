#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

void getMemoryInfo() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    // Total physical memory (in bytes)
    unsigned long long totalMemory = memInfo.ullTotalPhys;
    // Free physical memory (in bytes)
    unsigned long long freeMemory = memInfo.ullAvailPhys;

    // Convert to MB for readability
    double totalMemoryMB = totalMemory / (1024.0 * 1024.0);
    double freeMemoryMB = freeMemory / (1024.0 * 1024.0);
    double usedMemoryMB = totalMemoryMB - freeMemoryMB;

    std::cout << "Total Memory: " << totalMemoryMB << " MB\n";
    std::cout << "Used Memory: " << usedMemoryMB << " MB\n";
    std::cout << "Free Memory: " << freeMemoryMB << " MB\n";

#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);

    // Total physical memory (in bytes)
    unsigned long long totalMemory = memInfo.totalram * memInfo.mem_unit;
    // Free physical memory (in bytes)
    unsigned long long freeMemory = memInfo.freeram * memInfo.mem_unit;

    // Convert to MB for readability
    double totalMemoryMB = totalMemory / (1024.0 * 1024.0);
    double freeMemoryMB = freeMemory / (1024.0 * 1024.0);
    double usedMemoryMB = totalMemoryMB - freeMemoryMB;

    std::cout << "Total Memory: " << totalMemoryMB << " MB\n";
    std::cout << "Used Memory: " << usedMemoryMB << " MB\n";
    std::cout << "Free Memory: " << freeMemoryMB << " MB\n";

#endif
}

int main() {
    std::cout << "System Memory Information:\n";
    getMemoryInfo();
    return 0;
}