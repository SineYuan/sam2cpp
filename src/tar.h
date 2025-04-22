#pragma once

#include <string>
#include <cstdint>

namespace sam2 {

struct TarEntry {
    std::string name;
    uint64_t size;
    const uint8_t* data;
    bool is_directory;
};

std::vector<TarEntry> parseTar(const uint8_t* buffer, size_t bufferSize);

} // namespace sam2
