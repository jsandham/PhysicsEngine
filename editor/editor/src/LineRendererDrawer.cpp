#include "../include/LineRendererDrawer.h"

#include "components/LineRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

LineRendererDrawer::LineRendererDrawer()
{

}

LineRendererDrawer::~LineRendererDrawer()
{

}

void LineRendererDrawer::render(Component* component)
{
	if (ImGui::TreeNode("LineRenderer")) {
		LineRenderer* lineRenderer = dynamic_cast<LineRenderer*>(component);

		float start[3];
		start[0] = lineRenderer->start.x;
		start[1] = lineRenderer->start.y;
		start[2] = lineRenderer->start.z;

		float end[3];
		end[0] = lineRenderer->end.x;
		end[1] = lineRenderer->end.y;
		end[2] = lineRenderer->end.z;

		ImGui::InputFloat3("Start", &start[0]);
		ImGui::InputFloat3("End", &end[0]);

		lineRenderer->start.x = start[0];
		lineRenderer->start.y = start[1];
		lineRenderer->start.z = start[2];

		lineRenderer->end.x = end[0];
		lineRenderer->end.y = end[1];
		lineRenderer->end.z = end[2];

		ImGui::TreePop();
	}
}