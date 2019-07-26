#include "../include/CameraDrawer.h"

#include "components/Camera.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

CameraDrawer::CameraDrawer()
{

}

CameraDrawer::~CameraDrawer()
{

}

void CameraDrawer::render(Component* component)
{
	if (ImGui::TreeNode("Camera"))
	{
		Camera* camera = dynamic_cast<Camera*>(component);

		float position[3];
		position[0] = camera->position.x;
		position[1] = camera->position.y;
		position[2] = camera->position.z;

		float front[3];
		front[0] = camera->front.x;
		front[1] = camera->front.y;
		front[2] = camera->front.z;

		float up[3];
		up[0] = camera->up.x;
		up[1] = camera->up.y;
		up[2] = camera->up.z;

		float backgroundColor[4];
		backgroundColor[0] = camera->backgroundColor.x;
		backgroundColor[1] = camera->backgroundColor.y;
		backgroundColor[2] = camera->backgroundColor.z;
		backgroundColor[3] = camera->backgroundColor.w;

		ImGui::InputFloat3("Position", &position[0]);
		ImGui::InputFloat3("Front", &front[0]);
		ImGui::InputFloat3("Up", &up[0]);
		ImGui::ColorEdit4("Background Color", &backgroundColor[0]);

		camera->position.x = position[0];
		camera->position.y = position[1];
		camera->position.z = position[2];

		camera->front.x = front[0];
		camera->front.y = front[1];
		camera->front.z = front[2];

		camera->up.x = up[0];
		camera->up.y = up[1];
		camera->up.z = up[2];

		camera->backgroundColor.x = backgroundColor[0];
		camera->backgroundColor.y = backgroundColor[1];
		camera->backgroundColor.z = backgroundColor[2];
		camera->backgroundColor.w = backgroundColor[3];

		if (ImGui::TreeNode("Viewport"))
		{
			ImGui::InputInt("x", &camera->viewport.x);
			ImGui::InputInt("y", &camera->viewport.y);
			ImGui::InputInt("Width", &camera->viewport.width);
			ImGui::InputInt("Height", &camera->viewport.height);

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Frustum"))
		{
			ImGui::InputFloat("Field of View", &camera->frustum.fov);
			ImGui::InputFloat("Near Plane", &camera->frustum.nearPlane);
			ImGui::InputFloat("Far Plane", &camera->frustum.farPlane);

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}