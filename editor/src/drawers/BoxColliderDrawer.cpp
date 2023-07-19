#include "../../include/drawers/BoxColliderDrawer.h"

#include "components/BoxCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{
}

BoxColliderDrawer::~BoxColliderDrawer()
{
}

void BoxColliderDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("BoxCollider", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::BoxCollider* boxCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::BoxCollider>(id);

		if (boxCollider != nullptr)
		{
			ImGui::Text(("ComponentId: " + id.toString()).c_str());

			if (ImGui::TreeNode("Bounds"))
			{
				glm::vec3 centre = boxCollider->mAABB.mCentre;
				glm::vec3 size = boxCollider->mAABB.mSize;

				if (ImGui::InputFloat3("Centre", glm::value_ptr(centre)))
				{
					boxCollider->mAABB.mCentre = centre;
				}
				if (ImGui::InputFloat3("Size", glm::value_ptr(size)))
				{
					boxCollider->mAABB.mSize = size;
				}

				ImGui::TreePop();
			}

			bool enabled = boxCollider->mEnabled;
			if (ImGui::Checkbox("Enabled?", &enabled))
			{
				boxCollider->mEnabled = enabled;
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
				PhysicsEngine::BoxCollider* boxCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::BoxCollider>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(boxCollider->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::BoxCollider>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool BoxColliderDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}