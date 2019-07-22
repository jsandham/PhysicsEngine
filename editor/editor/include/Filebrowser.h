#ifndef __FILEBROWSER_H__
#define __FILEBROWSER_H__

#include <iostream>
#include <sstream>
#include <direct.h>
#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

namespace PhysicsEditor
{
	typedef enum FilebrowserMode
	{
		Open,
		Save
	};

	class Filebrowser
	{
		private:
			std::string currentDirectory;    // current directory the file browser is in
			std::vector<std::string> currentFiles;    // files located in the current directory
			std::vector<std::string> currentDirectories;    // directories located in the current directory
			std::vector<std::string> currentDirectoryShortPaths;    // current directory path short names
			std::vector<std::string> currentDirectoryLongPaths;    // current directory path long names

			bool isVisible;
			FilebrowserMode mode;
			std::vector<char> inputBuffer;
			std::string currentFilter;

		public:
			Filebrowser();
			~Filebrowser();

			void render(bool becomeVisibleThisFrame);
			void renderOpenMode();
			void renderSaveMode();
			void setMode(FilebrowserMode mode);

		private:
			void renderOpen();
			void renderSave();
			bool BeginFilterDropdown(std::string filter);
			void EndFilterDropdown();
	};


	static std::vector<std::string> split(const std::string& s, char delim) {
		std::vector<std::string> elems;
		elems.reserve(10);
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
		return elems;
	}

	static std::vector<std::string> getDirectoryLongPaths(const std::string path)
	{
		std::string temp = path;
		std::vector<std::string> directories;
		directories.reserve(10);
		while (temp.length() > 0){
			directories.push_back(temp);

			size_t index = 0;
			for (size_t i = temp.length() - 1; i > 0; i--){
				if (temp[i] == '\\'){
					index = i;
					break;
				}
			}

			temp = temp.substr(0, index);
		}

		return directories;
	}

	static std::string currentWorkingDirectory()
	{
		char* cwd = _getcwd(0, 0); // **** microsoft specific ****
		std::string working_directory(cwd);
		std::free(cwd);
		return working_directory;
	}

	static std::vector<std::string> getDirectoryContents(const std::string &directory, int contentType )
	{
		std::vector<std::string> files;

		HANDLE dir;
		WIN32_FIND_DATAA file_data;

		if ((dir = FindFirstFileA((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
			return files; /* No files found */

		do {
			const std::string file_name = file_data.cFileName;
			const std::string full_file_name = directory + "/" + file_name;
			const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

			if (file_name[0] == '.')
				continue;

			if (contentType == 0){ // directories only
				if (!is_directory){
					continue;
				}
			}
			else if (contentType == 1){ // files only
				if (is_directory){
					continue;
				}
			}

			//out.push_back(full_file_name);
			files.push_back(full_file_name);
		} while (FindNextFileA(dir, &file_data));

		FindClose(dir);

		return files;
	}

	static std::vector<std::string> getFilesInDirectory(const std::string &directory)
	{
		return getDirectoryContents(directory, 1);
	}

	static std::vector<std::string> getDirectoriesInDirectory(const std::string &directory)
	{
		return getDirectoryContents(directory, 0);
	}

	static std::vector<std::string> getFilesInDirectory(const std::string &directory, std::string extension)
	{
		std::vector<std::string> files;
		std::string search_path = directory + "/*.*";
		WIN32_FIND_DATAA fd;
		HANDLE hFind = ::FindFirstFileA(search_path.c_str(), &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
	
					std::string file = fd.cFileName;
					if (file.substr(file.find_last_of(".") + 1) == extension) {
						files.push_back(directory + file);
					}
				}
			} while (::FindNextFileA(hFind, &fd));
			::FindClose(hFind);
		}
		return files;
	}
}

#endif