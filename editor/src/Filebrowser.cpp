#define NOMINMAX

#include <algorithm>

#include "../include/FileBrowser.h"
#include "../include/FileSystemUtil.h"

#include "imgui.h"

#include "../include/imgui/imgui_extensions.h"

namespace fs = std::filesystem;

using namespace PhysicsEditor;

Filebrowser::Filebrowser()
{
    mIsVisible = false;
    mOpenClicked = false;
    mSaveClicked = false;
    mSelectFolderClicked = false;
    mMode = FilebrowserMode::Open;

    mCurrentDirectoryPath = fs::current_path();

    mInputBuffer.resize(256);

    mCurrentFilter = ".";

    mOpenFile = fs::path();
    mSaveFile = fs::path();
    mSelectedFolder = fs::path();
}

Filebrowser::~Filebrowser()
{
}

void Filebrowser::render(const std::filesystem::path& cwd, bool becomeVisibleThisFrame)
{
    mOpenClicked = false;
    mSaveClicked = false;
    mSelectFolderClicked = false;

    if (mIsVisible != becomeVisibleThisFrame)
    {
        mIsVisible = becomeVisibleThisFrame;
        if (becomeVisibleThisFrame)
        {
            mCurrentDirectoryPath = cwd;

            std::error_code error_code;
            for (const fs::directory_entry& entry : fs::directory_iterator(mCurrentDirectoryPath, error_code))
            {
                if (fs::is_directory(entry, error_code))
                {
                    mCurrentDirectories.push_back(entry.path());
                }
                else if (fs::is_regular_file(entry, error_code))
                {
                    mCurrentFiles.push_back(entry.path());
                }
            }

            ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 400.0f), ImVec2(1920.0f, 1080.0f));
            ImGui::OpenPopup("Filebrowser");
        }
    }

    if (ImGui::BeginPopupModal("Filebrowser", nullptr, ImGuiWindowFlags_NoResize))
    {
        float windowWidth = ImGui::GetWindowWidth();

        ImGui::Text(mCurrentDirectoryPath.string().c_str());
        ImGui::Text(mOpenFile.string().c_str());
        ImGui::Text(mSaveFile.string().c_str());
        ImGui::Text(mSelectedFolder.string().c_str());

        std::vector<std::string> directoryNamesInCurrentDirectoryPath =
            PhysicsEditor::split(mCurrentDirectoryPath, '\\');
       
        std::vector<std::filesystem::path> directoryPathsInCurrentDirectoryPath =
            PhysicsEditor::getDirectoryPaths(mCurrentDirectoryPath);

        for (size_t i = 0; i < directoryNamesInCurrentDirectoryPath.size(); i++)
        {
            std::filesystem::path directoryPath =
                directoryPathsInCurrentDirectoryPath[directoryPathsInCurrentDirectoryPath.size() - i - 1];

            if (ImGui::Button(directoryNamesInCurrentDirectoryPath[i].c_str()))
            {
                mCurrentDirectoryPath = directoryPath;

                mCurrentFiles.clear();
                mCurrentDirectories.clear();

                std::error_code error_code;
                for (const fs::directory_entry& entry : fs::directory_iterator(mCurrentDirectoryPath, error_code))
                {
                    if (fs::is_directory(entry, error_code))
                    {
                        mCurrentDirectories.push_back(entry.path());
                    }
                    else if (fs::is_regular_file(entry, error_code))
                    {
                        mCurrentFiles.push_back(entry.path());
                    }
                }

                directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(mCurrentDirectoryPath, '\\');
                directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(mCurrentDirectoryPath);

                if (mMode == FilebrowserMode::SelectFolder)
                {
                    for (int j = 0; j < std::min(256, (int)directoryNamesInCurrentDirectoryPath[i].length()); j++)
                    {
                        mInputBuffer[j] = directoryNamesInCurrentDirectoryPath[i][j];
                    }
                    for (int j = std::min(256, (int)directoryNamesInCurrentDirectoryPath[i].length()); j < 256; j++)
                    {
                        mInputBuffer[j] = '\0';
                    }
                }
            }

            std::vector<std::string> directories;
            std::error_code error_code;
            for (const auto& dir : fs::directory_iterator(directoryPath, error_code))
            {
                directories.push_back(dir.path().filename().string());
            }

            if (directories.size() > 0)
            {
                ImGui::SameLine(0, 0);

                int s = -1;
                if (ImGui::BeginDropdown(">##" + std::to_string(i), directories, &s))
                {
                    if (s >= 0)
                    {
                        mCurrentDirectoryPath = directoryPath / directories[s];

                        mCurrentFiles.clear();
                        mCurrentDirectories.clear();

                        std::error_code error_code;
                        for (const fs::directory_entry& entry : fs::directory_iterator(mCurrentDirectoryPath, error_code))
                        {
                            if (fs::is_directory(entry, error_code))
                            {
                                mCurrentDirectories.push_back(entry.path());
                            }
                            else if (fs::is_regular_file(entry, error_code))
                            {
                                mCurrentFiles.push_back(entry.path());
                            }
                        }
                        
                        directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(mCurrentDirectoryPath, '\\');
                        directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(mCurrentDirectoryPath);

                        if (mMode == FilebrowserMode::SelectFolder)
                        {
                            for (int j = 0; j < std::min(256, (int)directories[s].length()); j++)
                            {
                                mInputBuffer[j] = directories[s][j];
                            }
                            for (int j = std::min(256, (int)directories[s].length()); j < 256; j++)
                            {
                                mInputBuffer[j] = '\0';
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

        ImGuiTextFilter textFilter(mCurrentFilter.c_str());
        std::vector<std::string> filteredCurrentFiles;
        for (size_t i = 0; i < mCurrentFiles.size(); i++)
        {
            if (textFilter.PassFilter(mCurrentFiles[i].filename().string().c_str()))
            {
                filteredCurrentFiles.push_back(mCurrentFiles[i].filename().string());
            }
        }

        if (filteredCurrentFiles.size() == 0)
        {
            filteredCurrentFiles.push_back("");
        }

        std::vector<FilebrowserItem> items;
        for (size_t i = 0; i < mCurrentDirectories.size(); i++)
        {
            FilebrowserItem item;
            item.name = mCurrentDirectories[i].filename().string();
            item.type = 0;
            item.selected = false;

            items.push_back(item);
        }

        if (mMode != FilebrowserMode::SelectFolder)
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
                        mCurrentDirectoryPath = mCurrentDirectoryPath / items[i].name;

                        mCurrentFiles.clear();
                        mCurrentDirectories.clear();
                        
                        std::error_code error_code;
                        for (const fs::directory_entry& entry : fs::directory_iterator(mCurrentDirectoryPath, error_code))
                        {
                            if (fs::is_directory(entry, error_code))
                            {
                                mCurrentDirectories.push_back(entry.path());
                            }
                            else if (fs::is_regular_file(entry, error_code))
                            {
                                mCurrentFiles.push_back(entry.path());
                            }
                        }
                        
                        directoryNamesInCurrentDirectoryPath = PhysicsEditor::split(mCurrentDirectoryPath, '\\');
                        directoryPathsInCurrentDirectoryPath = PhysicsEditor::getDirectoryPaths(mCurrentDirectoryPath);
                    }
                }

                if (items[i].type == 1 || mMode == FilebrowserMode::SelectFolder)
                {
                    for (int j = 0; j < std::min(256, (int)items[selection].name.length()); j++)
                    {
                        mInputBuffer[j] = items[selection].name[j];
                    }
                    for (int j = std::min(256, (int)items[selection].name.length()); j < 256; j++)
                    {
                        mInputBuffer[j] = '\0';
                    }
                }
            }
        }
        ImGui::ListBoxFooter();
        ImGui::PopItemWidth();

        if (mMode == FilebrowserMode::Open)
        {
            renderOpenMode();
        }
        else if (mMode == FilebrowserMode::Save)
        {
            renderSaveMode();
        }
        else if (mMode == FilebrowserMode::SelectFolder)
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
    if (ImGui::InputText("##File Name", &mInputBuffer[0], (int)mInputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }
    ImGui::SameLine();
    const char *filterNames[] = {"Text Files (.txt)",  "Obj Files (.obj)", "Scene Files (.scene)",
                                 "JSON Files (.yaml)", "All Files (*)",    "IniFiles (.ini)"};
    const char *filters[] = {".txt", ".obj", ".scene", ".yaml", ".", ".ini"};
    static int filterIndex = 4;
    ImGui::SetNextItemWidth(filterDropDownWidth);
    ImGui::Combo("##Filter", &filterIndex, filterNames, IM_ARRAYSIZE(filterNames));
    mCurrentFilter = filters[filterIndex];

    int index = 0;
    for (size_t i = 0; i < mInputBuffer.size(); i++)
    {
        if (mInputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }
    mOpenFile = mCurrentDirectoryPath / std::string(mInputBuffer.begin(), mInputBuffer.begin() + index);

    if (ImGui::Button("Open"))
    {
        mOpenClicked = true;
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
    if (ImGui::InputText("##File Name", &mInputBuffer[0], (int)mInputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }

    const char* filterNames[] = { "Scene Files (.scene)", "All Files (*)" };
    const char* filters[] = { ".scene", "."};
    static int filterIndex = 1;

    ImGui::SetNextItemWidth(saveAsTypeTitleWidth);
    ImGui::Text("Save As Type");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(saveAsTypeDropDownWidth);
    ImGui::Combo("##Filter", &filterIndex, filterNames, IM_ARRAYSIZE(filterNames));
    mCurrentFilter = filters[filterIndex];

    int index = 0;
    for (size_t i = 0; i < mInputBuffer.size(); i++)
    {
        if (mInputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }

    switch (filterIndex)
    {
    case 0:
        mSaveFile = mCurrentDirectoryPath / (std::string(mInputBuffer.begin(), mInputBuffer.begin() + index) + ".scene");
        break;
    default:
        mSaveFile = mCurrentDirectoryPath / std::string(mInputBuffer.begin(), mInputBuffer.begin() + index);
        break;
    }

    if (ImGui::Button("Save"))
    {
        mSaveClicked = true;
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
    if (ImGui::InputText("##File Name", &mInputBuffer[0], (int)mInputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }

    int index = 0;
    for (size_t i = 0; i < mInputBuffer.size(); i++)
    {
        if (mInputBuffer[i] == '\0')
        {
            index = (int)i;
            break;
        }
    }
    mSelectedFolder = std::string(mInputBuffer.begin(), mInputBuffer.begin() + index);

    if (ImGui::Button("Select Folder"))
    {
        mSelectFolderClicked = true;

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
    mMode = mode;
}

std::filesystem::path Filebrowser::getOpenFilePath() const
{
    return mOpenFile;
}

std::filesystem::path Filebrowser::getSaveFilePath() const
{
    return mSaveFile;
}

std::filesystem::path Filebrowser::getSelectedFolderPath() const
{
    if (mSelectedFolder.empty() || mCurrentDirectoryPath.filename() == mSelectedFolder)
    {
        return mCurrentDirectoryPath;
    }

    return mCurrentDirectoryPath / mSelectedFolder;
}

std::filesystem::path Filebrowser::getCurrentDirectoryPath() const
{
    return mCurrentDirectoryPath;
}

bool Filebrowser::isOpenClicked() const
{
    return mOpenClicked;
}

bool Filebrowser::isSaveClicked() const
{
    return mSaveClicked;
}

bool Filebrowser::isSelectFolderClicked() const
{
    return mSelectFolderClicked;
}