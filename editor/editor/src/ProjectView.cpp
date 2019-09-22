#include "../include/ProjectView.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

ProjectView::ProjectView()
{

}

ProjectView::~ProjectView()
{

}

void ProjectView::render(bool isOpenedThisFrame)
{
	static bool projectViewActive = true;

	if (isOpenedThisFrame) {
		projectViewActive = isOpenedThisFrame;
	}

	if (!projectViewActive) {
		return;
	}

	if (ImGui::Begin("Project View", &projectViewActive)) {

	}

	ImGui::End();
}