#ifndef PREFERENCES_WINDOW_H__
#define PREFERENCES_WINDOW_H__

#include "../EditorClipboard.h"
#include "imgui.h"

namespace ImGui
{
    void StyleColorsDracula(ImGuiStyle* dst = NULL);
    void StyleColorsCherry(ImGuiStyle* dst = NULL);
    void StyleColorsLightGreen(ImGuiStyle* dst = NULL);
    void StyleColorsYellow(ImGuiStyle* dst = NULL);
    void StyleColorsGrey(ImGuiStyle* dst = NULL);
    void StyleColorsCharcoal(ImGuiStyle* dst = NULL);
    void StyleColorsCorporate(ImGuiStyle* dst = NULL);
} // namespace ImGui

namespace PhysicsEditor
{
    enum class EditorStyle
    {
        Classic,
        Light,
        Dark,
        Dracula,
        Cherry,
        LightGreen,
        Yellow,
        Grey,
        Charcoal,
        Corporate,
        Count
    };

class PreferencesWindow
{
  private:
    bool mOpen;

  public:
    PreferencesWindow();
    ~PreferencesWindow();
    PreferencesWindow(const PreferencesWindow &other) = delete;
    PreferencesWindow &operator=(const PreferencesWindow &other) = delete;

    void init(Clipboard &clipboard);
    void update(Clipboard& clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif
