#include "../../include/drawers/TransformDrawer.h"

#include "components/Transform.h"
#include "components/ComponentTypes.h"

#include "imgui.h"

using namespace PhysicsEditor;

TransformDrawer::TransformDrawer()
{
}

TransformDrawer::~TransformDrawer()
{
}

void TransformDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("Transform", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::Transform* transform = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Transform>(id);

		if (transform != nullptr)
		{
			ImGui::Text(("ComponentId: " + transform->getGuid().toString()).c_str());

			glm::vec3 position = transform->getPosition();
			glm::quat rotation = transform->getRotation();
			glm::vec3 scale = transform->getScale();
			glm::vec3 eulerAngles = glm::degrees(glm::eulerAngles(rotation));

			if (ImGui::DragFloat3("Position", glm::value_ptr(position)))
			{
				transform->setPosition(position);
			}

			if (ImGui::DragFloat3("Rotation", glm::value_ptr(eulerAngles)))
			{
				glm::quat x = glm::angleAxis(glm::radians(eulerAngles.x), glm::vec3(1.0f, 0.0f, 0.0f));
				glm::quat y = glm::angleAxis(glm::radians(eulerAngles.y), glm::vec3(0.0f, 1.0f, 0.0f));
				glm::quat z = glm::angleAxis(glm::radians(eulerAngles.z), glm::vec3(0.0f, 0.0f, 1.0f));

				transform->setRotation(z * y * x);
			}
			if (ImGui::DragFloat3("Scale", glm::value_ptr(scale)))
			{
				transform->setScale(scale);
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
				PhysicsEngine::Transform* transform = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Transform>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(transform->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::BoxCollider>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool TransformDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}