#ifndef __FILEBROWSER_H__
#define __FILEBROWSER_H__

#include <direct.h>
#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

namespace PhysicsEditor
{
	class Filebrowser
	{
		private:
			std::string currentPath;
			std::vector<std::string> currentFiles;

			bool wasVisible;
			char inputBuf[256];
			std::vector<char> inputBuffer;
			std::string currentFilter;

		public:
			Filebrowser();
			~Filebrowser();

			void render(bool isVisible);


			bool BeginFilterDropdown(std::string filter);
			void EndFilterDropdown();


			//bool BeginButtonDropDown(const char* label, ImVec2 buttonSize);
			//void EndButtonDropDown();

	};




	static std::string currentWorkingDirectory()
	{
		char* cwd = _getcwd(0, 0); // **** microsoft specific ****
		std::string working_directory(cwd);
		std::free(cwd);
		return working_directory;
	}

	static std::vector<std::string> getFilesInDirectory(const std::string &directory)
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

			if (is_directory)
				continue;

			//out.push_back(full_file_name);
			files.push_back(full_file_name);
		} while (FindNextFileA(dir, &file_data));

		FindClose(dir);

		return files;
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

	/*static std::vector<std::string> getDirectories()
	{
		std::vector<std::string> directories;
		std::string search_path = ""
	}*/
}



#endif