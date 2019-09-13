#include <iostream>
#include <sstream>
#include <direct.h>
#include <Windows.h>
#include <memory>
#include <vector>
#include <stack>
#include <string>
#include <fileapi.h>

#include "../include/FileSystemUtil.h"

using namespace PhysicsEditor;

bool PhysicsEditor::doesFileExist(std::string filePath)
{
	if (GetFileAttributesA(filePath.c_str()) == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND)
	{
		return false;
	}

	return true;
}

bool PhysicsEditor::doesDirectoryExist(const std::string directoryPath)
{
	DWORD ftyp = GetFileAttributesA(directoryPath.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
	{
		return false;
	}

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
	{
		return true;
	}

	return false;
}

bool PhysicsEditor::createDirectory(std::string path)
{
	//if (CreateDirectoryA(path.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	//{
	//	// CopyFile(...)
	//}
	//else
	//{
	//	// Failed to create directory.
	//}
	return CreateDirectoryA(path.c_str(), NULL);
}

bool PhysicsEditor::deleteDirectory(std::string path)
{
	//static const std::string allFilesMask(L"\\*");

	WIN32_FIND_DATAA findData;

	// First, delete the contents of the directory, recursively for subdirectories
	std::string searchMask = path + "\\";
	HANDLE searchHandle = ::FindFirstFileExA(searchMask.c_str(), FindExInfoBasic, &findData, FindExSearchNameMatch, nullptr, 0);
	if (searchHandle == INVALID_HANDLE_VALUE) {
		DWORD lastError = ::GetLastError();
		if (lastError != ERROR_FILE_NOT_FOUND) { // or ERROR_NO_MORE_FILES, ERROR_NOT_FOUND?
			return false;
			//throw std::runtime_error("Could not start directory enumeration");
		}
	}

	// Did this directory have any contents? If so, delete them first
	if (searchHandle != INVALID_HANDLE_VALUE) {
		//SearchHandleScope scope(searchHandle);
		for (;;) {

			// Do not process the obligatory '.' and '..' directories
			if (findData.cFileName[0] != '.') {
				bool isDirectory =
					((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) ||
					((findData.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);

				// Subdirectories need to be handled by deleting their contents first
				std::string filePath = path + "\\" + findData.cFileName;
				if (isDirectory) {
					if (deleteDirectory(filePath) == false) {
						return false;
					}
				}
				else {
					BOOL result = ::DeleteFileA(filePath.c_str());
					if (result == FALSE) {
						return false;
						//throw std::runtime_error("Could not delete file");
					}
				}
			}

			// Advance to the next file in the directory
			BOOL result = ::FindNextFileA(searchHandle, &findData);
			if (result == FALSE) {
				DWORD lastError = ::GetLastError();
				if (lastError != ERROR_NO_MORE_FILES) {
					return false;
					//throw std::runtime_error("Error enumerating directory");
				}
				break; // All directory contents enumerated and deleted
			}

		} // for
	}

	// The directory is empty, we can now safely remove it
	BOOL result = ::RemoveDirectoryA(path.c_str());
	if (result == FALSE) {
		return false;
		//throw std::runtime_error("Could not remove directory");
	}

	return true;
}

std::vector<std::string> PhysicsEditor::split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	elems.reserve(10);
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> PhysicsEditor::getDirectoryPaths(const std::string path)
{
	std::string temp = path;
	std::vector<std::string> directories;
	directories.reserve(10);
	while (temp.length() > 0) {
		directories.push_back(temp);

		size_t index = 0;
		for (size_t i = temp.length() - 1; i > 0; i--) {
			if (temp[i] == '\\') {
				index = i;
				break;
			}
		}

		temp = temp.substr(0, index);
	}

	return directories;
}

std::string PhysicsEditor::currentWorkingDirectoryPath()
{
	char* cwd = _getcwd(0, 0); // **** microsoft specific ****
	std::string working_directory(cwd);
	std::free(cwd);
	return working_directory;
}

std::vector<std::string> PhysicsEditor::getDirectoryContents(const std::string& path, int contentType, bool returnFullPaths)
{
	std::vector<std::string> files;

	HANDLE dir;
	WIN32_FIND_DATAA file_data;

	if ((dir = FindFirstFileA((path + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
		return files; /* No files found */

	do {
		const std::string file_name = file_data.cFileName;
		const std::string full_file_name = path + "\\" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (contentType == 0) { // directories only
			if (!is_directory) {
				continue;
			}
		}
		else if (contentType == 1) { // files only
			if (is_directory) {
				continue;
			}
		}

		if (returnFullPaths) {
			files.push_back(full_file_name);
		}
		else{
			files.push_back(file_name);
		}
	} while (FindNextFileA(dir, &file_data));

	FindClose(dir);

	return files;
}

std::vector<std::string> PhysicsEditor::getDirectoriesInDirectory(const std::string& path, bool returnFullPaths)
{
	return getDirectoryContents(path, 0, returnFullPaths);
}

std::vector<std::string> PhysicsEditor::getFilesInDirectory(const std::string& path, bool returnFullPaths)
{
	return getDirectoryContents(path, 1, returnFullPaths);
}

std::vector<std::string> PhysicsEditor::getFilesInDirectory(const std::string& path, std::string extension, bool returnFullPaths)
{
	std::vector<std::string> files;
	std::string search_path = path + "/*.*";
	WIN32_FIND_DATAA fd;
	HANDLE hFind = ::FindFirstFileA(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {

				std::string file = fd.cFileName;
				if (file.substr(file.find_last_of(".") + 1) == extension) {
					files.push_back(path + file);
				}
			}
		} while (::FindNextFileA(hFind, &fd));
		::FindClose(hFind);
	}
	return files;
}

std::vector<std::string> PhysicsEditor::getFilesInDirectoryRecursive(const std::string& path, bool returnFullPaths)
{
	std::vector<std::string> files;

	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATAA fd;
	std::string spec;
	std::stack<std::string> directories;

	directories.push(path);

	std::string currentPath = path;

	while (!directories.empty()) {
		currentPath = directories.top();
		spec = currentPath + "\\*";
		directories.pop();

		hFind = FindFirstFileA(spec.c_str(), &fd);
		if (hFind == INVALID_HANDLE_VALUE) {
			return files;
		}

		do {
			if (strcmp(fd.cFileName, ".") != 0 &&
				strcmp(fd.cFileName, "..") != 0) {
				if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					directories.push(currentPath + "\\" + fd.cFileName);
				}
				else {
					files.push_back(currentPath + "\\" + fd.cFileName);
				}
			}
		} while (FindNextFileA(hFind, &fd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES) {
			FindClose(hFind);
			return files;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
	}

	return files;
}