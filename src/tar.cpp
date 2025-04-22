
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "tar.h"


namespace sam2 {

struct TarHeader {
    char name[100];     // 文件名
    char mode[8];       // 权限
    char uid[8];        // 用户ID
    char gid[8];        // 组ID
    char size[12];      // 文件大小（八进制ASCII）
    char mtime[12];     // 修改时间
    char checksum[8];   // 校验和
    char typeflag;      // 文件类型（0-普通文件，5-目录等）
    char linkname[100]; // 链接目标
    char magic[6];      // 标识符（如"ustar"）
    // 其他字段省略...
};

std::vector<TarEntry> parseTar(const uint8_t* buffer, size_t bufferSize) {
    std::vector<TarEntry> entries;
    size_t offset = 0;

    while (offset + 512 <= bufferSize) {
        const TarHeader* header = reinterpret_cast<const TarHeader*>(buffer + offset);

        // 检查结束标记
        bool isEnd = true;
        for (int i = 0; i < 512; ++i) {
            if (buffer[offset + i] != 0) {
                isEnd = false;
                break;
            }
        }
        if (isEnd) break;

        // 提取文件名（处理终止符）
        std::string name(header->name, strnlen(header->name, 100));

        // 转换文件大小（八进制字符串转十进制）
        uint64_t size = strtoull(header->size, nullptr, 8);

        // 计算数据块位置和大小
        uint64_t dataOffset = offset + 512;
        uint64_t blockSize = ((size + 511) / 512) * 512;

        // 检查边界
        if (dataOffset + blockSize > bufferSize) break;

        // 提取条目信息
        TarEntry entry;
        entry.name = name;
        entry.size = size;
        entry.data = buffer + dataOffset;
        entry.is_directory = (header->typeflag == '5');

        // 仅处理普通文件和目录
        if (header->typeflag == '0' || header->typeflag == '\0' || entry.is_directory) {
            entries.push_back(entry);
        }

        // 移动到下一个头块
        offset = dataOffset + blockSize;
    }

    return entries;
}

} // namespace sam2
