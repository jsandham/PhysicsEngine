#include "../include/MeshRendererDrawer.h"

#include "components/MeshRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{

}

MeshRendererDrawer::~MeshRendererDrawer()
{

}

void MeshRendererDrawer::render(Component* component)
{
	if(ImGui::TreeNode("MeshRenderer"))
	{
		MeshRenderer* meshRenderer = dynamic_cast<MeshRenderer*>(component);

		//Guid meshId;
		//Guid materialIds[8];

		ImGui::Checkbox("Is Static?", &meshRenderer->isStatic);

		ImGui::TreePop();
	}
}