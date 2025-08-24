#include <include/hub/downloader.h>
#include <curl/curl.h>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sys/stat.h> // For mkdir

// A simple C-style callback for libcurl to write data to a file
size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

// Helper to create a directory if it doesn't exist
void create_directory_if_not_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

namespace xinfer::hub {

// Implementation for list_models would involve a GET request to an API endpoint
// and parsing a JSON response. This is a placeholder for that logic.
std::vector<ModelInfo> list_models(const std::string& hub_url) {
    std::cout << "Warning: hub::list_models() is not fully implemented yet." << std::endl;
    // TODO: Perform HTTP GET to hub_url/v1/models
    // TODO: Parse the JSON response into a vector of ModelInfo
    return {};
}

std::string download_engine(const std::string& model_id,
                          const HardwareTarget& target,
                          const std::string& cache_dir,
                          const std::string& hub_url)
{
    create_directory_if_not_exists(cache_dir);

    // Construct a unique filename for the cached engine
    std::string filename = model_id + "_" +
                           target.gpu_architecture + "_" +
                           target.tensorrt_version + "_" +
                           target.precision + ".engine";
    std::string local_path = cache_dir + "/" + filename;

    // If the file already exists in the cache, just return the path
    if (std::ifstream(local_path).good()) {
        std::cout << "Found engine in cache: " << local_path << std::endl;
        return local_path;
    }

    // Construct the download URL
    std::string download_url = hub_url + "/v1/download/" + filename;

    std::cout << "Downloading engine from: " << download_url << std::endl;

    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl.");
    }

    FILE* fp = fopen(local_path.c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to open file for writing: " + local_path);
    }

    curl_easy_setopt(curl, CURLOPT_URL, download_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L); // Fail on HTTP errors >= 400
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    CURLcode res = curl_easy_perform(curl);

    fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        // If download failed, remove the potentially empty file
        remove(local_path.c_str());
        throw std::runtime_error("Failed to download engine file. curl error: " + std::string(curl_easy_strerror(res)));
    }

    std::cout << "Successfully downloaded and cached engine to: " << local_path << std::endl;
    return local_path;
}

} // namespace xinfer::hub