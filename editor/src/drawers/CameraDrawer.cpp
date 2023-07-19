#include "../../include/drawers/CameraDrawer.h"

#include "components/Camera.h"

#include "imgui.h"

using namespace PhysicsEditor;

CameraDrawer::CameraDrawer()
{
}

CameraDrawer::~CameraDrawer()
{
}

void CameraDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::Camera* camera = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Camera>(id);

		if (camera != nullptr)
		{
			ImGui::Text(("ComponentId: " + id.toString()).c_str());

			int renderPath = static_cast<int>(camera->mRenderPath);
			int colorTarget = static_cast<int>(camera->mColorTarget);
			int mode = static_cast<int>(camera->mMode);
			int ssao = static_cast<int>(camera->mSSAO);

			const char* renderPathNames[] = { "Forward", "Deferred" };
			const char* colorTargetNames[] = { "Color", "Normal", "Position", "Linear Depth", "Shadow Cascades" };
			const char* modeNames[] = { "Main", "Secondary" };
			const char* ssaoNames[] = { "On", "Off" };

			if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2))
			{
				camera->mRenderPath = static_cast<PhysicsEngine::RenderPath>(renderPath);
			}

			if (ImGui::Combo("Color Target", &colorTarget, colorTargetNames, 5))
			{
				camera->mColorTarget = static_cast<PhysicsEngine::ColorTarget>(colorTarget);
			}

			if (ImGui::Combo("Mode", &mode, modeNames, 2))
			{
				camera->mMode = static_cast<PhysicsEngine::CameraMode>(mode);
			}

			if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
			{
				camera->mSSAO = static_cast<PhysicsEngine::CameraSSAO>(ssao);
			}

			/*Guid renderTargetId = camera->mRenderTextureId;

			std::string renderTargetName = "None (Render Texture)";
			if (renderTargetId.isValid())
			{
				renderTargetName = renderTargetId.toString();
			}

			bool releaseTriggered = false;
			bool clearClicked = false;
			bool isClicked = ImGui::Slot("Render Target", renderTargetName, &releaseTriggered, &clearClicked);

			if (releaseTriggered && clipboard.getDraggedType() == InteractionType::RenderTexture)
			{
				renderTargetId = clipboard.getDraggedId();
				clipboard.clearDraggedItem();

				camera->mRenderTextureId = renderTargetId;
			}

			if (isClicked)
			{
				clipboard.setSelectedItem(InteractionType::RenderTexture, renderTargetId);
			}*/

			glm::vec4 backgroundColor = glm::vec4(camera->mBackgroundColor.mR, camera->mBackgroundColor.mG,
				camera->mBackgroundColor.mB, camera->mBackgroundColor.mA);

			if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor)))
			{
				camera->mBackgroundColor = PhysicsEngine::Color(backgroundColor);
			}

			if (ImGui::TreeNode("Viewport"))
			{
				int x = camera->getViewport().mX;
				int y = camera->getViewport().mY;
				int width = camera->getViewport().mWidth;
				int height = camera->getViewport().mHeight;

				if (ImGui::InputInt("x", &x))
				{
					camera->setViewport(x, y, width, height);
				}
				if (ImGui::InputInt("y", &y))
				{
					camera->setViewport(x, y, width, height);
				}
				if (ImGui::InputInt("Width", &width))
				{
					camera->setViewport(x, y, width, height);
				}
				if (ImGui::InputInt("Height", &height))
				{
					camera->setViewport(x, y, width, height);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Frustum"))
			{
				float fov = camera->getFrustum().mFov;
				float nearPlane = camera->getFrustum().mNearPlane;
				float farPlane = camera->getFrustum().mFarPlane;

				if (ImGui::InputFloat("Field of View", &fov))
				{
					camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
				}
				if (ImGui::InputFloat("Near Plane", &nearPlane))
				{
					camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
				}
				if (ImGui::InputFloat("Far Plane", &farPlane))
				{
					camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
				}

				ImGui::TreePop();
			}

			// Directional light cascade splits
			int cascadeType = static_cast<int>(camera->mShadowCascades);

			const char* cascadeTypeNames[] = { "No Cascades", "Two Cascades", "Three Cascades", "Four Cascades", "Five Cascades" };

			if (ImGui::Combo("Shadow Cascades", &cascadeType, cascadeTypeNames, 5))
			{
				camera->mShadowCascades = static_cast<PhysicsEngine::ShadowCascades>(cascadeType);
			}

			if (camera->mShadowCascades != PhysicsEngine::ShadowCascades::NoCascades)
			{
				ImColor colors[5] = { ImColor(1.0f, 0.0f, 0.0f),
									  ImColor(0.0f, 1.0f, 0.0f),
									  ImColor(0.0f, 0.0f, 1.0f),
									  ImColor(0.0f, 1.0f, 1.0f),
									  ImColor(0.6f, 0.0f, 0.6f) };

				std::array<int, 5> splits = camera->getCascadeSplits();
				for (size_t i = 0; i < splits.size(); i++)
				{
					ImGui::PushItemWidth(0.125f * ImGui::GetWindowSize().x);

					ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;

					if (i <= static_cast<int>(camera->mShadowCascades))
					{
						ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)colors[i]);
						ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)colors[i]);
						ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)colors[i]);
					}
					else
					{
						ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
						ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
						ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));

						flags |= ImGuiInputTextFlags_ReadOnly;
					}

					if (ImGui::InputInt(("##Cascade Splits" + std::to_string(i)).c_str(), &splits[i], 0, 100, flags))
					{
						camera->setCascadeSplit(i, splits[i]);
					}

					ImGui::PopStyleColor(3);
					ImGui::PopItemWidth();
					ImGui::SameLine();
				}
				ImGui::Text("Cascade Splits");
			}

			bool enabled = camera->mEnabled;
			if (ImGui::Checkbox("Enabled?", &enabled))
			{
				camera->mEnabled = enabled;
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
				PhysicsEngine::Camera* camera = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Camera>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(camera->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::Camera>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool CameraDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}