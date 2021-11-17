#include "../../include/views/PreferencesWindow.h"
#include "../../include/imgui/imgui_styles.h"

#include "imgui.h"

using namespace PhysicsEditor;

PreferencesWindow::PreferencesWindow() : PopupWindow("##Preferences...", 600.0f, 300.0f, 400.0f, 400.0f)
{
}

PreferencesWindow::~PreferencesWindow()
{
}

void PreferencesWindow::init(Clipboard &clipboard)
{
}

void PreferencesWindow::update(Clipboard &clipboard)
{
    // Order matters here
    const char* themeNames[] = { "Classic",    "Light",  "Dark", "Dracula",  "Cherry",
                                "LightGreen", "Yellow", "Grey", "Charcoal", "Corporate" };

    if (ImGui::BeginCombo("##Themes", themeNames[0]))
    {
        for (int n = 0; n < 10; n++)
        {
            bool is_selected = false; // (currentTextureName == themeNames[n]); // You can store your selection
                                      // however you want, outside or inside your objects
            if (ImGui::Selectable(themeNames[n], is_selected))
            {
                if (n == 0)
                {
                    ImGui::StyleColorsClassic();
                }
                else if (n == 1)
                {
                    ImGui::StyleColorsLight();
                }
                else if (n == 2)
                {
                    ImGui::StyleColorsDark();
                }
                else if (n == 3)
                {
                    ImGui::StyleColorsDracula();
                }
                else if (n == 4)
                {
                    ImGui::StyleColorsCherry();
                }
                else if (n == 5)
                {
                    ImGui::StyleColorsLightGreen();
                }
                else if (n == 6)
                {
                    ImGui::StyleColorsYellow();
                }
                else if (n == 7)
                {
                    ImGui::StyleColorsGrey();
                }
                else if (n == 8)
                {
                    ImGui::StyleColorsCharcoal();
                }
                else if (n == 9)
                {
                    ImGui::StyleColorsCorporate();
                }
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();

    if (ImGui::Button("Ok"))
    {
        ImGui::CloseCurrentPopup();
    }
}