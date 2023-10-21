#include "../../include/drawers/MeshRendererDrawer.h"
#include "../../include/ProjectDatabase.h"

#include "components/MeshRenderer.h"
#include "components/ComponentTypes.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{
}

MeshRendererDrawer::~MeshRendererDrawer()
{
}

void MeshRendererDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::MeshRenderer* meshRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::MeshRenderer>(id);

		if (meshRenderer != nullptr)
		{
			ImGui::Text(("ComponentId: " + id.toString()).c_str());

			// Mesh
			{
				PhysicsEngine::Mesh* mesh = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Mesh>(meshRenderer->getMesh());

				ImVec2 windowSize = ImGui::GetWindowSize();
				windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

				if (ImGui::ButtonEx((mesh == nullptr ? "None (Mesh)" : mesh->mName).c_str(), ImVec2(windowSize.x, 0)))
				{
					clipboard.setSelectedItem(InteractionType::Mesh, mesh == nullptr ? PhysicsEngine::Guid::INVALID : mesh->getGuid());
				}

				if (ImGui::BeginDragDropTarget())
				{
					const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MESH_PATH");
					if (payload != nullptr)
					{
						const char* data = static_cast<const char*>(payload->Data);

						meshRenderer->setMesh(ProjectDatabase::getGuid(data));
					}
					ImGui::EndDragDropTarget();
				}

				ImVec2 size = ImGui::GetItemRectSize();
				ImVec2 position = ImGui::GetItemRectMin();

				ImVec2 topLeft = position;
				ImVec2 topRight = ImVec2(position.x + size.x, position.y);
				ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
				ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);

				ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
				ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
				ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
				ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);

				size.x += position.x;
				size.y += position.y;

				if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
				{
					ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
				}

				ImGui::SameLine();
				ImGui::Text("Mesh");
			}

			// Materials
			int materialCount = meshRenderer->mMaterialCount;
			const int increment = 1;
			ImGui::PushItemWidth(80);
			if (ImGui::InputScalar("Material Count", ImGuiDataType_S32, &materialCount, &increment, NULL, "%d"))
			{
				materialCount = std::max(0, std::min(materialCount, 8));

				meshRenderer->mMaterialCount = materialCount;
			}
			ImGui::PopItemWidth();

			PhysicsEngine::Guid materialIds[8];
			for (int i = 0; i < materialCount; i++)
			{
				materialIds[i] = meshRenderer->getMaterial(i);

				PhysicsEngine::Material* material = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Material>(materialIds[i]);

				ImVec2 windowSize = ImGui::GetWindowSize();
				windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

				std::string materialLabel = (material == nullptr) ? "None (Material)" : (material->mName + "##" + std::to_string(i));

				if (ImGui::ButtonEx(materialLabel.c_str(), ImVec2(windowSize.x, 0)))
				{
					if (material != nullptr)
					{
						clipboard.setSelectedItem(InteractionType::Material, material->getGuid());
					}
				}

				if (ImGui::BeginDragDropTarget())
				{
					const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MATERIAL_PATH");
					if (payload != nullptr)
					{
						const char* data = static_cast<const char*>(payload->Data);

						meshRenderer->setMaterial(ProjectDatabase::getGuid(data), i);
					}
					ImGui::EndDragDropTarget();
				}

				ImVec2 size = ImGui::GetItemRectSize();
				ImVec2 position = ImGui::GetItemRectMin();

				ImVec2 topLeft = position;
				ImVec2 topRight = ImVec2(position.x + size.x, position.y);
				ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
				ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);

				ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
				ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
				ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
				ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);

				size.x += position.x;
				size.y += position.y;

				if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
				{
					ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
				}

				ImGui::SameLine();
				ImGui::Text("Material");
			}

			bool isStatic = meshRenderer->mIsStatic;
			if (ImGui::Checkbox("Is Static?", &isStatic))
			{
				meshRenderer->mIsStatic = isStatic;
			}

			bool enabled = meshRenderer->mEnabled;
			if (ImGui::Checkbox("Enabled?", &enabled))
			{
				meshRenderer->mEnabled = enabled;
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
				PhysicsEngine::MeshRenderer* meshRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::MeshRenderer>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(meshRenderer->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::MeshRenderer>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool MeshRendererDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}