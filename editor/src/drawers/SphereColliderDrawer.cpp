#include "../../include/drawers/SphereColliderDrawer.h"

#include "components/SphereCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{
}

SphereColliderDrawer::~SphereColliderDrawer()
{
}

void SphereColliderDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("SphereCollider", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::SphereCollider* sphereCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::SphereCollider>(id);

		if (sphereCollider != nullptr)
		{
			ImGui::Text(("ComponentId: " + id.toString()).c_str());

			if (ImGui::TreeNode("Sphere"))
			{
				glm::vec3 centre = sphereCollider->mSphere.mCentre;
				float radius = sphereCollider->mSphere.mRadius;

				if (ImGui::InputFloat3("Centre", glm::value_ptr(centre)))
				{
					sphereCollider->mSphere.mCentre = centre;
				}
				if (ImGui::InputFloat("Radius", &radius))
				{
					sphereCollider->mSphere.mRadius = radius;
				}

				ImGui::TreePop();
			}

			bool enabled = sphereCollider->mEnabled;
			if (ImGui::Checkbox("Enabled?", &enabled))
			{
				sphereCollider->mEnabled = enabled;
			}
		}

		ImGui::TreePop();
	}

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();

	if (isHovered())
	{
		if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
		{
			if (ImGui::MenuItem("RemoveComponent", NULL, false, true))
			{
				PhysicsEngine::SphereCollider* sphereCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::SphereCollider>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(sphereCollider->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::SphereCollider>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool SphereColliderDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}