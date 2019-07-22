#include "../include/Hierarchy.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy()
{

}

Hierarchy::~Hierarchy()
{

}

void Hierarchy::render(bool isOpenedThisFrame)
{
	static bool hierarchyActive = true;

	if (isOpenedThisFrame){
		hierarchyActive = true;
	}

	if (!hierarchyActive){
		return;
	}

	if (ImGui::Begin("Hierarchy", &hierarchyActive))
	{








		ImGuiIO& io = ImGui::GetIO();

		ImGui::Text("Mouse clicked:");  
		if (ImGui::IsMouseClicked(1))
		{
			ImGui::SameLine(); ImGui::Text("b%d", 1);
		}

		ImGui::End();
	}
}