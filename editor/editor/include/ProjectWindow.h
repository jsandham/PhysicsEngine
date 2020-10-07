#ifndef __PROJECT_WINDOW_H__
#define __PROJECT_WINDOW_H__

#include "Filebrowser.h"

namespace PhysicsEditor
{
typedef enum ProjectWindowMode
{
    OpenProject,
    NewProject
} ProjectWindowMode;

class ProjectWindow
{
  private:
    bool isVisible;
    bool openClicked;
    bool createClicked;
    ProjectWindowMode mode;
    Filebrowser filebrowser;
    std::vector<char> inputBuffer;

  public:
    ProjectWindow();
    ~ProjectWindow();

    void render(bool becomeVisibleThisFrame);
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
