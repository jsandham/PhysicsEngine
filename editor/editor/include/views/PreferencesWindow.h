#ifndef __PREFERENCES_WINDOW_H__
#define __PREFERENCES_WINDOW_H__

#include "Window.h"

namespace PhysicsEditor
{
class PreferencesWindow : public Window
{
  public:
    PreferencesWindow();
    ~PreferencesWindow();
    PreferencesWindow(const PreferencesWindow& other) = delete;
    PreferencesWindow& operator=(const PreferencesWindow& other) = delete;

    void init(EditorClipboard& clipboard);
    void update(EditorClipboard& clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif
