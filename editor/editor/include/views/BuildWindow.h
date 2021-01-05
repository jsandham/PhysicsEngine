#ifndef __BUILD_WINDOW_H__
#define __BUILD_WINDOW_H__

#include "Window.h"

namespace PhysicsEditor
{
class BuildWindow : public Window
{
  public:
    BuildWindow();
    ~BuildWindow();
    BuildWindow(const BuildWindow &other) = delete;
    BuildWindow &operator=(const BuildWindow &other) = delete;

    void init(EditorClipboard &clipboard);
    void update(EditorClipboard &clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif
