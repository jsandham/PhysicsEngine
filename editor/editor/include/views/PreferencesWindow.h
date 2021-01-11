#ifndef __PREFERENCES_WINDOW_H__
#define __PREFERENCES_WINDOW_H__

#include "PopupWindow.h"

namespace PhysicsEditor
{
class PreferencesWindow : public PopupWindow
{
  public:
    PreferencesWindow();
    ~PreferencesWindow();
    PreferencesWindow(const PreferencesWindow &other) = delete;
    PreferencesWindow &operator=(const PreferencesWindow &other) = delete;

    void init(EditorClipboard &clipboard) override;
    void update(EditorClipboard &clipboard) override;
};
} // namespace PhysicsEditor

#endif
