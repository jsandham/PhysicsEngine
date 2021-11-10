#ifndef __FILESYSTEM_UTIL_H__
#define __FILESYSTEM_UTIL_H__

#include <string>
#include <vector>
#include <filesystem>

namespace PhysicsEditor
{
std::vector<std::string> split(const std::filesystem::path path, char delim);
std::vector<std::filesystem::path> getDirectoryPaths(const std::filesystem::path path);
} // namespace PhysicsEditor

#endif