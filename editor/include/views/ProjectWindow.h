#ifndef PROJECT_WINDOW_H__
#define PROJECT_WINDOW_H__

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class ProjectWindow
{
  private:
    std::vector<char> mInputBuffer;
    std::string mSelectedFolder;
    bool mOpen;

  public:
    ProjectWindow();
    ~ProjectWindow();
    ProjectWindow(const ProjectWindow &other) = delete;
    ProjectWindow &operator=(const ProjectWindow &other) = delete;

    void init(Clipboard &clipboard);
    void update(Clipboard& clipboard, bool isOpenedThisFrame);

    std::string getProjectName() const;
};
} // namespace PhysicsEditor
#endif
