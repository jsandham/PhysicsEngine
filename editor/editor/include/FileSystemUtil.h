#ifndef __FILESYSTEM_UTIL_H__
#define __FILESYSTEM_UTIL_H__

#include <string>
#include <vector>

namespace PhysicsEditor
{
	bool createDirectory(std::string path);
	std::vector<std::string> split(const std::string& s, char delim);
	std::vector<std::string> getDirectoryPaths(const std::string path);
	std::string currentWorkingDirectoryPath();
	std::vector<std::string> getDirectoryContents(const std::string& path, int contentType, bool returnFullPaths = false);
	std::vector<std::string> getFilesInDirectory(const std::string& path, bool returnFullPaths = false);
	std::vector<std::string> getDirectoriesInDirectory(const std::string& path, bool returnFullPaths = false);
	std::vector<std::string> getFilesInDirectory(const std::string& path, std::string extension, bool returnFullPaths = false);
}

#endif