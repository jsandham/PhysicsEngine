#ifndef __PROJECT_WINDOW_H__
#define __PROJECT_WINDOW_H__

#include "PopupWindow.h"

#include "../Filebrowser.h"

namespace PhysicsEditor
{
enum class ProjectWindowMode
{
    OpenProject,
    NewProject
};

class ProjectWindow : public PopupWindow
{
  private:
    ProjectWindowMode mode;
    Filebrowser filebrowser;
    std::vector<char> inputBuffer;

  public:
    ProjectWindow();
    ~ProjectWindow();
    ProjectWindow(const ProjectWindow &other) = delete;
    ProjectWindow &operator=(const ProjectWindow &other) = delete;

    void init(EditorClipboard &clipboard) override;
    void update(EditorClipboard &clipboard) override;

    void setMode(ProjectWindowMode mode);

    std::string getProjectName() const;
    std::string getSelectedFolderPath() const;

  private:
    void renderOpenMode(EditorClipboard& clipboard);
    void renderNewMode(EditorClipboard& clipboard);
};
} // namespace PhysicsEditor
#endif
