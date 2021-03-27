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

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

    void setMode(ProjectWindowMode mode);

    std::string getProjectName() const;
    std::string getSelectedFolderPath() const;

  private:
    void renderOpenMode(Clipboard& clipboard);
    void renderNewMode(Clipboard& clipboard);
};
} // namespace PhysicsEditor
#endif
