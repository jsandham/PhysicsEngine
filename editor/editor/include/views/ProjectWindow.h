#ifndef __PROJECT_WINDOW_H__
#define __PROJECT_WINDOW_H__

#include "Window.h"

#include "../Filebrowser.h"

namespace PhysicsEditor
{
enum class ProjectWindowMode
{
    OpenProject,
    NewProject
};

class ProjectWindow : public Window
{
  private:
    bool openClicked;
    bool createClicked;
    ProjectWindowMode mode;
    Filebrowser filebrowser;
    std::vector<char> inputBuffer;

  public:
    ProjectWindow();
    ~ProjectWindow();
    ProjectWindow(const ProjectWindow &other) = delete;
    ProjectWindow &operator=(const ProjectWindow &other) = delete;

    void init(EditorClipboard &clipboard);
    void update(EditorClipboard &clipboard, bool isOpenedThisFrame);

    void setMode(ProjectWindowMode mode);

    bool isOpenClicked() const;
    bool isCreateClicked() const;

    std::string getProjectName() const;
    std::string getSelectedFolderPath() const;

  private:
    void renderOpenMode();
    void renderNewMode();
};
} // namespace PhysicsEditor
#endif
