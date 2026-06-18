#pragma once

enum class Device {
    CPU,
    CUDA
};

// struct Device {
//     DeviceType type;
//     int index;

//     Device(DeviceType t = DeviceType::CPU, int idx = 0) : type(t), index(idx) {}

//     bool operator==(const Device& other) const {
//         return type == other.type && index == other.index;
//     }

//     bool operator!=(const Device& other) const {
//         return !(*this == other);
//     }
// };