#ifndef PROJECT_WINDOW_H__
#define PROJECT_WINDOW_H__

#include "PopupWindow.h"

namespace PhysicsEditor
{
class ProjectWindow : public PopupWindow
{
  private:
    std::vector<char> mInputBuffer;
    std::string mSelectedFolder;

  public:
    ProjectWindow();
    ~ProjectWindow();
    ProjectWindow(const ProjectWindow &other) = delete;
    ProjectWindow &operator=(const ProjectWindow &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

    std::string getProjectName() const;
};
} // namespace PhysicsEditor
#endif
