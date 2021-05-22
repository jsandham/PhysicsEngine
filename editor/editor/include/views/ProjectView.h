#ifndef __PROJECT_H__
#define __PROJECT_H__

#include <string>
#include <vector>
#include <filesystem>

#include "Window.h"

#include "../ProjectTree.h"
#include "../EditorClipboard.h"

#include "imgui.h"

namespace PhysicsEditor
{
class ProjectView : public Window
{
  private:
    ProjectTree mProjectTree;
    ProjectNode *mSelected;
    //std::string mRightPanelSelectedPath;
    std::filesystem::path mRightPanelSelectedPath;

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
