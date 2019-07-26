#include "../include/RigidbodyDrawer.h"

#include "components/Rigidbody.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

RigidbodyDrawer::RigidbodyDrawer()
{

}

RigidbodyDrawer::~RigidbodyDrawer()
{

}

void RigidbodyDrawer::render(Component* component)
{
	if (ImGui::TreeNode("Rigidbody"))
	{
		Rigidbody* rigidbody = dynamic_cast<Rigidbody*>(component);

		bool useGravity;
		float mass;
		float drag;
		float angularDrag;

		ImGui::InputCheck("Use Gravity", &rigidbody->useGravity);
	}
}