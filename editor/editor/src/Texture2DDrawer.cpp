#include "../include/Texture2DDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Texture2D.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Texture2DDrawer::Texture2DDrawer()
{

}

Texture2DDrawer::~Texture2DDrawer()
{

}

void Texture2DDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	Texture2D* texture = world->getAsset<Texture2D>(id);

	ImGui::Separator();

	// Draw texture child window
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
		ImGui::BeginChild("DrawTextureWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);
	
		ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
		ImGui::Button("RGB");
		ImGui::PopStyleColor();

		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
		ImGui::Button("R");
		ImGui::PopStyleColor();

		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
		ImGui::Button("G");
		ImGui::PopStyleColor();

		ImGui::SameLine();
		
		ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
		ImGui::Button("B");
		ImGui::PopStyleColor();

		ImGui::Image((void*)(intptr_t)texture->tex, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));

		ImGui::EndChild();
	}
}