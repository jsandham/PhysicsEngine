#include "../include/SphereColliderDrawer.h"

#include "components/SphereCollider.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{

}

SphereColliderDrawer::~SphereColliderDrawer()
{

}

void SphereColliderDrawer::render(Component* component)
{
	if (ImGui::TreeNode("SphereCollider")) 
	{
		SphereCollider* sphereCollider = dynamic_cast<SphereCollider*>(component);

		if (ImGui::TreeNode("Sphere")) {
			float centre[3];
			centre[0] = sphereCollider->sphere.centre.x;
			centre[1] = sphereCollider->sphere.centre.y;
			centre[2] = sphereCollider->sphere.centre.z;

			ImGui::InputFloat3("Centre", &centre[0]);
			ImGui::InputFloat("Radius", &sphereCollider->sphere.radius);

			sphereCollider->sphere.centre.x = centre[0];
			sphereCollider->sphere.centre.y = centre[1];
			sphereCollider->sphere.centre.z = centre[2];

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}