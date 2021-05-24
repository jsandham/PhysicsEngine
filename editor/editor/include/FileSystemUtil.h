#ifndef __FILESYSTEM_UTIL_H__
#define __FILESYSTEM_UTIL_H__

#include <string>
#include <vector>

namespace PhysicsEditor
{
//bool doesFileExist(std::string filePath);
//bool doesDirectoryExist(std::string directoryPath);
//bool createDirectory(std::string path);
//bool deleteDirectory(std::string path);
//bool deleteFile(std::string path);
////bool getFileTime(std::string path, std::string &createTime, std::string &accessTime, std::string &writeTime);
////std::string getDirectoryName(std::string path);
////std::string getFileName(std::string path);
//std::string getFileExtension(std::string path);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> getDirectoryPaths(const std::string path);
//std::string currentWorkingDirectoryPath();
std::vector<std::string> getDirectoryContents(const std::string &path, int contentType, bool returnFullPaths = false);
std::vector<std::string> getDirectoriesInDirectory(const std::string &path, bool returnFullPaths = false);
//std::vector<std::string> getFilesInDirectory(const std::string &path, bool returnFullPaths = false);
//std::vector<std::string> getFilesInDirectory(const std::string &path, std::string extension,
//                                             bool returnFullPaths = false);
//std::vector<std::string> getFilesInDirectoryRecursive(const std::string &path, bool returnFullPaths = false);
} // namespace PhysicsEditor

#endif