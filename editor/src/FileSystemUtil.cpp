#include "../include/FileSystemUtil.h"

using namespace PhysicsEditor;

std::vector<std::string> PhysicsEditor::split(const std::filesystem::path path, char delim)
{
    std::vector<std::string> temp;
    for (auto& part : path)
    {
        if (part != "\\")
        {
            temp.push_back(part.string());
        }
    }

    return temp;
}

std::vector<std::filesystem::path> PhysicsEditor::getDirectoryPaths(const std::filesystem::path path)
{
    std::vector<std::filesystem::path> directoryPaths;

    std::filesystem::path temp = path;
    while (temp != temp.root_path())
    {
        directoryPaths.push_back(temp);

        temp = temp.parent_path();
    }

    directoryPaths.push_back(temp);

    return directoryPaths;
}