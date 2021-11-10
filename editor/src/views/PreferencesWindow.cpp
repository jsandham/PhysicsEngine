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
    float windowWidth = ImGui::GetWindowWidth();

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
                if (themeNames[n] == "Classic")
                {
                    ImGui::StyleColorsClassic();
                }
                else if (themeNames[n] == "Light")
                {
                    ImGui::StyleColorsLight();
                }
                else if (themeNames[n] == "Dark")
                {
                    ImGui::StyleColorsDark();
                }
                else if (themeNames[n] == "Dracula")
                {
                    ImGui::StyleColorsDracula();
                }
                else if (themeNames[n] == "Cherry")
                {
                    ImGui::StyleColorsCherry();
                }
                else if (themeNames[n] == "LightGreen")
                {
                    ImGui::StyleColorsLightGreen();
                }
                else if (themeNames[n] == "Yellow")
                {
                    ImGui::StyleColorsYellow();
                }
                else if (themeNames[n] == "Grey")
                {
                    ImGui::StyleColorsGrey();
                }
                else if (themeNames[n] == "Charcoal")
                {
                    ImGui::StyleColorsCharcoal();
                }
                else if (themeNames[n] == "Corporate")
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