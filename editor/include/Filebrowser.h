#ifndef __FILEBROWSER_H__
#define __FILEBROWSER_H__

#include <Windows.h>
#include <direct.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace PhysicsEditor
{
enum class FilebrowserMode
{
    Open,
    Save,
    SelectFolder
};

struct FilebrowserItem
{
    std::string name;
    int type;
    bool selected;
};

class Filebrowser
{
  private:
    std::filesystem::path mCurrentDirectoryPath;            // current Path the file browser is in
    std::vector<std::filesystem::path> mCurrentFiles;       // files located in the current directory path
    std::vector<std::filesystem::path> mCurrentDirectories; // directories located in the current directory path

    std::filesystem::path mOpenFile;
    std::filesystem::path mSaveFile;
    std::filesystem::path mSelectedFolder;

    bool mIsVisible;
    bool mOpenClicked;
    bool mSaveClicked;
    bool mSelectFolderClicked;
    FilebrowserMode mMode;
    std::vector<char> mInputBuffer;
    std::string mCurrentFilter;

  public:
    Filebrowser();
    ~Filebrowser();

    void render(const std::filesystem::path& cwd, bool becomeVisibleThisFrame);
    void setMode(FilebrowserMode mode);
    std::filesystem::path getOpenFilePath() const;
    std::filesystem::path getSaveFilePath() const;
    std::filesystem::path getSelectedFolderPath() const;
    std::filesystem::path getCurrentDirectoryPath() const;
    bool isOpenClicked() const;
    bool isSaveClicked() const;
    bool isSelectFolderClicked() const;

  private:
    void renderOpenMode();
    void renderSaveMode();
    void renderSelectFolderMode();
};
} // namespace PhysicsEditor

#endif