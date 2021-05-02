#define NOMINMAX

#include <algorithm>

#include "../include/FileBrowser.h"
#include "../include/FileSystemUtil.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

Filebrowser::Filebrowser()
{
    isVisible = false;
    openClicked = false;
    saveClicked = false;
    selectFolderClicked = false;
    mode = FilebrowserMode::Open;

    currentDirectoryPath = PhysicsEditor::currentWorkingDirectoryPath();

    inputBuffer.resize(256);

    currentFilter = ".";

    openFile = "";
    saveFile = "";
    selectedFolder = "";
}

Filebrowser::~Filebrowser()
{
}

void Filebrowser::render(std::string cwd, bool becomeVisibleThisFrame)
{
    openClicked = false;
    saveClicked = false;
    selectFolderClicked = false;

    if (isVisible != becomeVisibleThisFrame)
    {
        isVisible = becomeVisibleThisFrame;
        if (becomeVisibleThisFrame)
        {
            currentDirectoryPath = cwd;
            currentFiles = PhysicsEditor::getFilesInDirectory(currentDirectoryPath);
            currentDirectories = PhysicsEditor::getDirectoriesInDirectory(currentDirectoryPath);

            ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 400.0f), ImVec2(1920.0f, 1080.0f));
            ImGui::OpenPopup("Filebrowser");
        }
    }

    if (ImGui::BeginPopupModal("Filebrowser", nullptr, ImGuiWindowFlags_NoResize))
    {
        float windowWidth = ImGui::GetWindowWidth();

        ImGui::Text(currentDirectoryPath.c_str());
        ImGui::Text(openFile.c_str());
        ImGui::Text(saveFile.c_str());
        ImGui::Text(selectedFolder.c_str());

        std::vector<std::string> directoryNamesInCurrentDirectoryPath =
            PhysicsEditor::split(currentDirectoryPath, '\\');
        std::vector<std::string> directoryPathsInCurrentDirectoryPath =
            PhysicsEditor::getDirectoryPaths(currentDirectoryPath);

        for (size_t i = 0; i < directoryNamesInCurrentDirectoryPath.size(); i++)
        {

            std::string directoryPath =
                directoryPathsInCurrentDirectoryPath[directoryPathsInCurrentDirectoryPath.size() - i - 1];

            if (ImGui::Button(directoryNamesInCurrentDirectoryPath[i].c_str()))
            {
                currentDirectoryPath = directoryPath;

                currentFiles = PhysicsEditor::getFilesInDirectory(currentDirectoryPath);
                currentDirectories = PhysicsEditor::getDirectoriesInDirectory(currentDirectoryPath);
                directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(currentDirectoryPath, '\\');
                directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(currentDirectoryPath);

                if (mode == FilebrowserMode::SelectFolder)
                {
                    for (int j = 0; j < std::min(256, (int)directoryNamesInCurrentDirectoryPath[i].length()); j++)
                    {
                        inputBuffer[j] = directoryNamesInCurrentDirectoryPath[i][j];
                    }
                    for (int j = std::min(256, (int)directoryNamesInCurrentDirectoryPath[i].length()); j < 256; j++)
                    {
                        inputBuffer[j] = '\0';
                    }
                }
            }

            std::vector<std::string> directories = PhysicsEditor::getDirectoriesInDirectory(directoryPath);
            std::vector<std::string> directoryPaths = PhysicsEditor::getDirectoriesInDirectory(directoryPath, true);

            if (directories.size() > 0)
            {
                ImGui::SameLine(0, 0);

                int s = -1;
                if (ImGui::BeginDropdown(">##" + std::to_string(i), directories, &s))
                {
                    if (s >= 0)
                    {
                        currentDirectoryPath = directoryPaths[s];

                        currentFiles = PhysicsEditor::getFilesInDirectory(currentDirectoryPath);
                        currentDirectories = PhysicsEditor::getDirectoriesInDirectory(currentDirectoryPath);
                        directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(currentDirectoryPath, '\\');
                        directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(currentDirectoryPath);

                        if (mode == FilebrowserMode::SelectFolder)
                        {
                            for (int j = 0; j < std::min(256, (int)directories[s].length()); j++)
                            {
                                inputBuffer[j] = directories[s][j];
                            }
                            for (int j = std::min(256, (int)directories[s].length()); j < 256; j++)
                            {
                                inputBuffer[j] = '\0';
                            }
                        }
                    }

                    ImGui::EndDropdown();
                }
            }

            if (i != directoryNamesInCurrentDirectoryPath.size() - 1)
            {
                ImGui::SameLine(0, 0);
            }
        }

        ImGuiTextFilter textFilter(currentFilter.c_str());
        std::vector<std::string> filteredCurrentFiles;
        for (size_t i = 0; i < currentFiles.size(); i++)
        {
            if (textFilter.PassFilter(currentFiles[i].c_str()))
            {
                filteredCurrentFiles.push_back(currentFiles[i]);
            }
        }

        if (filteredCurrentFiles.size() == 0)
        {
            filteredCurrentFiles.push_back("");
        }

        std::vector<FilebrowserItem> items;
        for (size_t i = 0; i < currentDirectories.size(); i++)
        {
            FilebrowserItem item;
            item.name = currentDirectories[i];
            item.type = 0;
            item.selected = false;

            items.push_back(item);
        }

        if (mode != FilebrowserMode::SelectFolder)
        {
            for (size_t i = 0; i < filteredCurrentFiles.size(); i++)
            {
                FilebrowserItem item;
                item.name = filteredCurrentFiles[i];
                item.type = 1;
                item.selected = false;

                items.push_back(item);
            }
        }

        static int selection = 0;
        ImGui::PushItemWidth(-1);
        ImGui::ListBoxHeader("##Current directory contents", (int)items.size(), 10);
        for (size_t i = 0; i < items.size(); i++)
        {
            if (i == selection)
            {
                items[i].selected = true;
            }

            if (ImGui::Selectable(items[i].name.c_str(), items[i].selected, ImGuiSelectableFlags_AllowDoubleClick))
            {
                selection = (int)i;

                if (items[i].type == 0)
                {
                    if (ImGui::IsMouseDoubleClicked(0))
                    {
                        currentDirectoryPath = currentDirectoryPath + "\\" + items[i].name;

                        currentFiles = PhysicsEditor::getFilesInDirectory(currentDirectoryPath);
                        currentDirectories = PhysicsEditor::getDirectoriesInDirectory(currentDirectoryPath);
                        directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(currentDirectoryPath, '\\');
                        directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(currentDirectoryPath);
                    }
                }

                if (items[i].type == 1 || mode == FilebrowserMode::SelectFolder)
                {
                    for (int j = 0; j < std::min(256, (int)items[selection].name.length()); j++)
                    {
                        inputBuffer[j] = items[selection].name[j];
                    }
                    for (int j = std::min(256, (int)items[selection].name.length()); j < 256; j++)
                    {
                        inputBuffer[j] = '\0';
                    }
                }
            }
        }
        ImGui::ListBoxFooter();
        ImGui::PopItemWidth();

        if (mode == FilebrowserMode::Open)
        {
            renderOpenMode();
        }
        else if (mode == FilebrowserMode::Save)
        {
            renderSaveMode();
        }
        else if (mode == FilebrowserMode::SelectFolder)
        {
            renderSelectFolderMode();
        }

        ImGui::EndPopup();
    }
}

void Filebrowser::renderOpenMode()
{
    float windowWidth = ImGui::GetWindowWidth();

    float fileNameTitleWidth = 80.0f;
    float filterDropDownWidth = 200.0f;
    float inputTextWidth = windowWidth - fileNameTitleWidth - filterDropDownWidth - 10.0f;

    ImGui::SetNextItemWidth(fileNameTitleWidth);
    ImGui::Text("File Name");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputTextWidth);
    if (ImGui::InputText("##File Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }
    ImGui::SameLine();
    const char *filterNames[] = {"Text Files (.txt)",  "Obj Files (.obj)", "Scene Files (.scene)",
                                 "JSON Files (.json)", "All Files (*)",    "IniFiles (.ini)"};
    const char *filters[] = {".txt", ".obj", ".scene", ".json", ".", ".ini"};
    static int filterIndex = 4;
    ImGui::SetNextItemWidth(filterDropDownWidth);
    ImGui::Combo("##Filter", &filterIndex, filterNames, IM_ARRAYSIZE(filterNames));
    currentFilter = filters[filterIndex];

    int index = 0;
    for (size_t i = 0; i < inputBuffer.size(); i++)
    {
        if (inputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }
    openFile = std::string(inputBuffer.begin(), inputBuffer.begin() + index);

    if (ImGui::Button("Open"))
    {
        openClicked = true;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
    {
        ImGui::CloseCurrentPopup();
    }
}

void Filebrowser::renderSaveMode()
{
    float windowWidth = ImGui::GetWindowWidth();

    float fileNameTitleWidth = 120.0f;
    float saveAsTypeTitleWidth = 120.0f;
    float inputTextWidth = windowWidth - fileNameTitleWidth - 10.0f;
    float saveAsTypeDropDownWidth = windowWidth - fileNameTitleWidth - 10.0f;

    ImGui::SetNextItemWidth(fileNameTitleWidth);
    ImGui::Text("File Name");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputTextWidth);
    if (ImGui::InputText("##File Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }

    const char *saveAsTypeNames[] = {"Scene Files (.scene)", "All Files (*)"};
    const char *saveAsTypes[] = {".scene", "."};

    static int saveAsTypeIndex = 4;

    ImGui::SetNextItemWidth(saveAsTypeTitleWidth);
    ImGui::Text("Save As Type");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(saveAsTypeDropDownWidth);
    ImGui::Combo("##Filter", &saveAsTypeIndex, saveAsTypeNames, IM_ARRAYSIZE(saveAsTypeNames));

    int index = 0;
    for (size_t i = 0; i < inputBuffer.size(); i++)
    {
        if (inputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }
    saveFile = std::string(inputBuffer.begin(), inputBuffer.begin() + index);

    if (ImGui::Button("Save"))
    {
        saveClicked = true;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
    {
        ImGui::CloseCurrentPopup();
    }
}

void Filebrowser::renderSelectFolderMode()
{
    float windowWidth = ImGui::GetWindowWidth();

    float folderTitleWidth = 120.0f;
    float inputTextWidth = windowWidth - folderTitleWidth - 10.0f;
    float saveAsTypeDropDownWidth = windowWidth - folderTitleWidth - 10.0f;

    ImGui::SetNextItemWidth(folderTitleWidth);
    ImGui::Text("Folder");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputTextWidth);
    if (ImGui::InputText("##File Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }

    int index = 0;
    for (size_t i = 0; i < inputBuffer.size(); i++)
    {
        if (inputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }
    selectedFolder = std::string(inputBuffer.begin(), inputBuffer.begin() + index);

    if (ImGui::Button("Select Folder"))
    {
        selectFolderClicked = true;

        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
    {
        ImGui::CloseCurrentPopup();
    }
}

void Filebrowser::setMode(FilebrowserMode mode)
{
    this->mode = mode;
}

std::string Filebrowser::getOpenFile() const
{
    return openFile;
}

std::string Filebrowser::getOpenFilePath() const
{
    return currentDirectoryPath + "\\" + openFile;
}

std::string Filebrowser::getSaveFile() const
{
    return saveFile;
}

std::string Filebrowser::getSaveFilePath() const
{
    return currentDirectoryPath + "\\" + saveFile;
}

std::string Filebrowser::getSelectedFolder() const
{
    return selectedFolder;
}

std::string Filebrowser::getSelectedFolderPath() const
{
    if (currentDirectoryPath.substr(currentDirectoryPath.find_last_of("/\\") + 1) == selectedFolder)
    {
        return currentDirectoryPath;
    }

    return currentDirectoryPath + "\\" + selectedFolder;
}

std::string Filebrowser::getCurrentDirectoryPath() const
{
    return currentDirectoryPath;
}

bool Filebrowser::isOpenClicked() const
{
    return openClicked;
}

bool Filebrowser::isSaveClicked() const
{
    return saveClicked;
}

bool Filebrowser::isSelectFolderClicked() const
{
    return selectFolderClicked;
}