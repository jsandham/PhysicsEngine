#ifndef PROJECT_VIEW_H__
#define PROJECT_VIEW_H__

#include <string>
#include <vector>
#include <filesystem>

#include "../ProjectTree.h"
#include "../EditorClipboard.h"

#include "imgui.h"

namespace PhysicsEditor
{
class ProjectView
{
  private:
    ProjectTree mProjectTree;

    InteractionType mHighlightedType;

    std::filesystem::path mHighlightedPath;
    std::filesystem::path mHoveredPath;
    std::filesystem::path mSelectedDirectoryPath;
    std::filesystem::path mSelectedFilePath;

    ImGuiTextFilter mFilter;

    bool mOpen;

  public:
    ProjectView();
    ~ProjectView();
    ProjectView(const ProjectView &other) = delete;
    ProjectView &operator=(const ProjectView &other) = delete;

    void init(Clipboard& clipboard);
    void update(Clipboard& clipboard, bool isOpenedThisFrame);

    void drawLeftPane();
    void drawRightPane(Clipboard &clipboard);

    void drawProjectTree();
    void drawProjectNodeRecursive(ProjectNode *node);
};
} // namespace PhysicsEditor

#endif
