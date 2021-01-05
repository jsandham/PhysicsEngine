#ifndef __FILEBROWSER_H__
#define __FILEBROWSER_H__

#include <Windows.h>
#include <direct.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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
    std::string currentDirectoryPath;            // current Path the file browser is in
    std::vector<std::string> currentFiles;       // files located in the current directory path
    std::vector<std::string> currentDirectories; // directories located in the current directory path

    bool isVisible;
    bool openClicked;
    bool saveClicked;
    bool selectFolderClicked;
    FilebrowserMode mode;
    std::vector<char> inputBuffer;
    std::string currentFilter;
    std::string openFile;
    std::string saveFile;
    std::string selectedFolder;

  public:
    Filebrowser();
    ~Filebrowser();

    void render(std::string cwd, bool becomeVisibleThisFrame);
    void setMode(FilebrowserMode mode);
    std::string getOpenFile() const;
    std::string getOpenFilePath() const;
    std::string getSaveFile() const;
    std::string getSaveFilePath() const;
    std::string getSelectedFolder() const;
    std::string getSelectedFolderPath() const;
    std::string getCurrentDirectoryPath() const;
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