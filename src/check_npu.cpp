#include <iostream>
#include <openvino/openvino.hpp>

int main() {
    try {
        // ov::Core 是 OpenVINO 的总入口。
        // 通过它可以查询当前系统里有哪些可用设备，以及把模型编译到指定设备上。
        ov::Core core;
        std::vector<std::string> available_devices = core.get_available_devices();

        std::cout << "[OpenVINO] 可用设备:" << std::endl;
        for (const auto& device : available_devices) {
            std::cout << "  - " << device << std::endl;
        }

        // 这里只做最简单的字符串判断：
        // 设备名里只要包含 "NPU"，就认为当前环境已经能看到 NPU 插件。
        bool has_npu = false;
        for (const auto& device : available_devices) {
            if (device.find("NPU") != std::string::npos) {
                has_npu = true;
                break;
            }
        }

        if (has_npu) {
            std::cout << "成功：检测到 NPU，可以使用。" << std::endl;
        } else {
            std::cout << "警告：设备列表里没有 NPU。" << std::endl;
            std::cout << "提示：请先 source oneAPI 和 OpenVINO 环境后再运行检查。"
                      << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "异常: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
