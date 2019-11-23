#include "../include/LibraryCache.h"
#include "../include/FileSystemUtil.h"

#include <fstream>
#include <string>
#include <vector>

#include "core/Log.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

LibraryCache::LibraryCache()
{

}

LibraryCache::~LibraryCache()
{

}

void LibraryCache::add(std::string filePath, FileInfo fileInfo)
{
	filePathToFileInfo.insert(std::pair<std::string, FileInfo>(filePath, fileInfo));
}

void LibraryCache::remove(std::string filePath)
{
	filePathToFileInfo.erase(filePath);
}

void LibraryCache::clear()
{
	filePathToFileInfo.clear();
}

bool LibraryCache::load(std::string libraryCachePath)
{
	Log::info("Loading library cache\n");
	std::fstream file(libraryCachePath, std::ios::in);

	bool error = false;
	if (file.is_open()) {
		std::vector<std::string> filePaths;

		std::string line;
		while (getline(file, line))
		{
			std::vector<std::string> splitLine = PhysicsEditor::split(line, ' ');
			if (splitLine.size() != 3) {
				Log::warn("Line in directory cache has incorrect format. Re-creating asset corresponding to this line in library instead of loading\n");
				continue;
			}

			std::string filePath = splitLine[0];
			std::string fileExtension = filePath.substr(filePath.find_last_of(".") + 1);
			std::string createTime = splitLine[1];
			std::string writeTime = splitLine[2];

			std::string infoMessage = "Loading: " + filePath + " " + createTime + " " + writeTime + "\n";
			Log::info(&infoMessage[0]);

			std::map<std::string, FileInfo>::iterator it = filePathToFileInfo.find(filePath);
			if (it == filePathToFileInfo.end()) {
				FileInfo fileInfo;
				fileInfo.filePath = filePath;
				fileInfo.fileExtension = fileExtension;
				fileInfo.createTime = createTime;
				fileInfo.writeTime = writeTime;

				filePathToFileInfo[filePath] = fileInfo;
			}
			else {
				Log::error("A duplicate entry was encountered when loading directory cache. Please delete library directory so it can be rebuilt\n");
				error = true;
				break;
			}
		}
		if (!file.eof())
		{
			error = true;
			Log::error("An end of file error was encountered when loading library from directory cache file. Please delete library directory so it can be rebuilt\n");
		}

		file.close();
	}
	else {
		Log::warn("Could not open library cache file when attempting to load. Library will be re-built instead of loading from cache file\n");
		return true;
	}

	if (error) {
		return false;
	}

	return true;
}

bool LibraryCache::save(std::string libraryCachePath)
{
	Log::info("Saving library cache\n");
	std::fstream file(libraryCachePath, std::ios::out);

	if (file.is_open()) {
		std::map<std::string, FileInfo>::iterator it = filePathToFileInfo.begin();
		for (it = filePathToFileInfo.begin(); it != filePathToFileInfo.end(); it++) {
			/*file << (it->first + " " + it->second.id.toString() + " " + it->second.createTime + " " + it->second.writeTime + "\n");*/
			file << (it->first + " " + it->second.createTime + " " + it->second.writeTime + "\n");
		}

		file.close();
	}
	else {
		Log::error("Could not open directory cache file for saving of library.\n");
		return false;
	}

	return true;
}

bool LibraryCache::contains(std::string filePath) const
{
	std::map<std::string, FileInfo>::const_iterator it = filePathToFileInfo.find(filePath);
	if (it != filePathToFileInfo.end()) {
		return true;
	}

	return false;
}

bool LibraryCache::isOutOfDate(std::string filePath, std::string createTime, std::string writeTime) const
{
	std::map<std::string, FileInfo>::const_iterator it = filePathToFileInfo.find(filePath);
	if (it != filePathToFileInfo.end()) {
		if (it->second.createTime != createTime || it->second.writeTime != writeTime) {
			return true;
		}
	}

	return false;
}

LibraryCache::iterator::iterator()
{

}

LibraryCache::iterator::~iterator()
{

}

bool LibraryCache::iterator::operator==(const iterator& other) const
{
	return it == other.it;
}

bool LibraryCache::iterator::operator!=(const iterator& other) const
{
	return it != other.it;
}

LibraryCache::iterator& LibraryCache::iterator::operator++()
{
	it++;
	return *this;
}

LibraryCache::iterator LibraryCache::iterator::operator++(int)
{
	iterator oldIter = *this;
	it++;
	return oldIter;
}

LibraryCache::iterator& LibraryCache::iterator::operator--()
{
	it--;
	return *this;
}

LibraryCache::iterator LibraryCache::iterator::operator--(int)
{
	iterator oldIter = *this;
	it--;
	return oldIter;
}

std::pair<const std::string, FileInfo>& LibraryCache::iterator::operator*() const
{
	return *it;
}

std::pair<const std::string, FileInfo>* LibraryCache::iterator::operator->() const
{
	return &(operator*());
}