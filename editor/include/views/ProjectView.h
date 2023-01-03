#ifndef PROJECT_VIEW_H__
#define PROJECT_VIEW_H__

#include <string>
#include <vector>
#include <filesystem>

#include "Window.h"

#include "../ProjectTree.h"

#include "imgui.h"

namespace PhysicsEditor
{
class ProjectView : public Window
{
  private:
    ProjectTree mProjectTree;

    InteractionType mHighlightedType;

    std::filesystem::path mHighlightedPath;
    std::filesystem::path mHoveredPath;
    std::filesystem::path mSelectedDirectoryPath;
    std::filesystem::path mSelectedFilePath;

    ImGuiTextFilter mFilter;

  public:
    ProjectView();
    ~ProjectView();
    ProjectView(const ProjectView &other) = delete;
    ProjectView &operator=(const ProjectView &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

    void drawLeftPane();
    void drawRightPane(Clipboard &clipboard);

    void drawProjectTree();
    void drawProjectNodeRecursive(ProjectNode *node);
};
} // namespace PhysicsEditor

#endif
