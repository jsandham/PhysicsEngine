#include "../include/SceneView.h"


#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

SceneView::SceneView()
{

}

SceneView::~SceneView()
{

}

void SceneView::render(GLuint mainTexture, bool isOpenedThisFrame)
{
	static bool sceneViewActive = true;

	if (isOpenedThisFrame) {
		sceneViewActive = true;
	}

	if (!sceneViewActive) {
		return;
	}

	ImGui::Begin("Scene View", &sceneViewActive);
	{
		//ImGui::Image((void*)(intptr_t)mainTexture, ImGui::GetContentRegionMax());

		ImVec2 windowOrigin = ImGui::GetWindowPos();
		ImVec2 contentSize = ImGui::GetWindowContentRegionMax();
		contentSize.y -= 30; // how do I get the header height of a window?

		ImGui::Image((void*)(intptr_t)mainTexture, contentSize);

		//ImVec2 p1 = windowOrigin;
		//ImVec2 p2 = ImVec2(p1.x + contentSize.x, p1.y + contentSize.y);
		//ImGui::GetWindowDrawList()->AddLine(p1, p2, IM_COL32(255, 0, 255, 255));
	}
	ImGui::End();
}