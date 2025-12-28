#pragma once
#include <string>
#include <vector>

namespace xinfer::deployer {

struct Device {
    std::string name;
    std::string ip;
    std::string user;
    std::string password; // Or use SSH keys
    std::string target_platform;
    std::string remote_path;
};

class IDeployer {
public:
    virtual ~IDeployer() = default;
    virtual bool connect(const Device& device) = 0;
    virtual void disconnect() = 0;

    /**
     * @brief Copies files to the remote device.
     * @param local_paths Vector of local file paths.
     * @return True if all files were sent successfully.
     */
    virtual bool send_files(const std::vector<std::string>& local_paths) = 0;

    /**
     * @brief Executes a command on the remote device.
     * @param command The shell command to run.
     * @param output Standard output from the command.
     * @return Exit code of the command.
     */
    virtual int execute(const std::string& command, std::string& output) = 0;
};

}