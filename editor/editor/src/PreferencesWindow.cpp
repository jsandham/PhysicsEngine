#include "../include/PreferencesWindow.h"
#include "../include/imgui_styles.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

PreferencesWindow::PreferencesWindow()
{
	isVisible = false;
}

PreferencesWindow::~PreferencesWindow()
{

}

void PreferencesWindow::render(bool becomeVisibleThisFrame)
{
	if (isVisible != becomeVisibleThisFrame) {
		isVisible = becomeVisibleThisFrame;
		if (becomeVisibleThisFrame) {
			//ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
			ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 200.0f), ImVec2(1920.0f, 1080.0f));
			//ImGui::SetNextWindowBgAlpha(1.0f);

			ImGui::OpenPopup("##Preferences...");
		}
	}

	bool preferencesWindowOpen = true;
	if (ImGui::BeginPopupModal("##Preferences...", &preferencesWindowOpen))
	{
		float windowWidth = ImGui::GetWindowWidth();

		const char* themeNames[] = {"Classic", "Light", "Dark", "Dracula", "Cherry", "LightGreen", "Yellow", "Grey", "Charcoal", "Corporate"};

		if (ImGui::BeginCombo("##Themes", themeNames[0]))
		{
			for (int n = 0; n < 10; n++)
			{
				bool is_selected = false;// (currentTextureName == themeNames[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(themeNames[n], is_selected)) {
					if (themeNames[n] == "Classic") {
						ImGui::StyleColorsClassic();
					}
					else if (themeNames[n] == "Light") {
						ImGui::StyleColorsLight();
					}
					else if (themeNames[n] == "Dark") {
						ImGui::StyleColorsDark();
					}
					else if (themeNames[n] == "Dracula") {
						ImGui::StyleColorsDracula();
					}
					else if (themeNames[n] == "Cherry") {
						ImGui::StyleColorsCherry();
					}
					else if (themeNames[n] == "LightGreen") {
						ImGui::StyleColorsLightGreen();
					}
					else if (themeNames[n] == "Yellow") {
						ImGui::StyleColorsYellow();
					}
					else if (themeNames[n] == "Grey") {
						ImGui::StyleColorsGrey();
					}
					else if (themeNames[n] == "Charcoal") {
						ImGui::StyleColorsCharcoal();
					}
					else if (themeNames[n] == "Corporate") {
						ImGui::StyleColorsCorporate();
					}
				}
			}
			ImGui::EndCombo();
		}
		ImGui::SameLine();

		
		if (ImGui::Button("Ok")) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
}