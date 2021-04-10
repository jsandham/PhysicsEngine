#ifndef __PROJECT_H__
#define __PROJECT_H__

#include <string>
#include <vector>

#include "Window.h"

#include "imgui.h"

namespace PhysicsEditor
{
struct ProjectNode
{
    ProjectNode *parent;
    std::vector<ProjectNode *> children;
    std::string directoryName;
    std::string directoryPath;
    std::vector<std::string> filePaths;
    int id;

    ProjectNode() : id(-1), parent(nullptr), directoryName(""), directoryPath("")
    {
    }


};

class ProjectView : public Window
{
  private:
    ProjectNode *root;
    ProjectNode *selected;
    std::vector<ProjectNode *> nodes;

    ImGuiTextFilter filter;

  public:
    ProjectView();
    ~ProjectView();
    ProjectView(const ProjectView &other) = delete;
    ProjectView &operator=(const ProjectView &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

    void drawLeftPane();
    void drawRightPane(Clipboard &clipboard);

    void deleteProjectTree();
    void buildProjectTree(const std::string &currentProjectPath);
    void drawProjectTree();
    void drawProjectNodeRecursive(ProjectNode *node);

    InteractionType getInteractionTypeFromFileExtension(const std::string extension);
};
} // namespace PhysicsEditor

#endif
