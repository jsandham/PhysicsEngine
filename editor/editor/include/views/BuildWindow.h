#ifndef __BUILD_WINDOW_H__
#define __BUILD_WINDOW_H__

#include "Window.h"

namespace PhysicsEditor
{
class BuildWindow
{
  public:
    BuildWindow();
    ~BuildWindow();
    BuildWindow(const BuildWindow &other) = delete;
    BuildWindow &operator=(const BuildWindow &other) = delete;

    void init(Clipboard &clipboard);
    void update(Clipboard &clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif
