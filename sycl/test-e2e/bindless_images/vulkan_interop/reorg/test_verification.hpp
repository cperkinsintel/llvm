#pragma once
#include <cmath>
#include <cstdlib>
#include <type_traits>

// Generates a deterministic test value based on position and channel
template <typename T>
T generateTestValue(size_t index, int channel, size_t rangeMax) {
    if constexpr (std::is_floating_point_v<T>) {
        // Float: 0.0 -> 1.0 gradient with channel offset
        float val = (float)index / (float)(rangeMax > 1 ? rangeMax - 1 : 1);
        return static_cast<T>(val + (float)channel * 0.1f);
    } else {
        // Integer: Wrapping pattern to avoid overflow
        return static_cast<T>((index + channel * 10) % 127); 
    }
}

// Compares values with appropriate tolerance for Floats
template <typename T>
bool checkValue(T actual, T expected) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(actual - expected) < 0.01f;
    } else {
        return actual == expected;
    }
}