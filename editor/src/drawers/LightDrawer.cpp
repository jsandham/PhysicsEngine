#include "../../include/drawers/LightDrawer.h"

#include "components/Light.h"
#include "components/ComponentTypes.h"

#include "imgui.h"

using namespace PhysicsEditor;

LightDrawer::LightDrawer()
{
}

LightDrawer::~LightDrawer()
{
}

void LightDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::Light* light = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Light>(id);

		if (light != nullptr)
		{
			ImGui::Text(("ComponentId: " + id.toString()).c_str());

			const char* lightTypes[] = { "Directional", "Spot", "Point" };
			int lightTypeIndex = static_cast<int>(light->mLightType);
			if (ImGui::Combo("Light Type", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes)))
			{
				light->mLightType = static_cast<PhysicsEngine::LightType>(lightTypeIndex);
			}

			glm::vec4 color = light->mColor;
			if (ImGui::ColorEdit4("Light Color", glm::value_ptr(color)))
			{
				light->mColor = color;
			}

			float intensity = light->mIntensity;
			if (ImGui::InputFloat("Intensity", &intensity))
			{
				light->mIntensity = std::max(0.0f, intensity);
			}

			if (light->mLightType == PhysicsEngine::LightType::Spot)
			{
				float spotAngleRad = glm::radians(light->mSpotAngle);
				float innerSpotAngleRad = glm::radians(light->mInnerSpotAngle);

				if (ImGui::SliderAngle("Spot Angle", &spotAngleRad, 0.0f, 90.0f))
				{
					light->mSpotAngle = glm::degrees(spotAngleRad);
					if (light->mSpotAngle < light->mInnerSpotAngle)
					{
						light->mInnerSpotAngle = light->mSpotAngle;
					}
				}
				if (ImGui::SliderAngle("Inner Spot Angle", &innerSpotAngleRad, 0.0f, 90.0f))
				{
					light->mInnerSpotAngle = glm::degrees(innerSpotAngleRad);
					if (light->mSpotAngle < light->mInnerSpotAngle)
					{
						light->mSpotAngle = light->mInnerSpotAngle;
					}
				}
			}

			const char* shadowTypes[] = { "Hard Shadows", "Soft Shadows", "No Shadows" };
			int shadowTypeIndex = static_cast<int>(light->mShadowType);
			if (ImGui::Combo("Shadow Type", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes)))
			{
				light->mShadowType = static_cast<PhysicsEngine::ShadowType>(shadowTypeIndex);
			}

			if (light->mShadowType != PhysicsEngine::ShadowType::None)
			{
				float shadowStrength = light->mShadowStrength;
				if (ImGui::SliderFloat("Shadow Strength", &shadowStrength, 0.0f, 1.0f))
				{
					light->mShadowStrength = shadowStrength;
				}

				float shadowBias = light->mShadowBias;
				if (ImGui::SliderFloat("Shadow Bias", &shadowBias, 0.0f, 0.1f))
				{
					light->mShadowBias = shadowBias;
				}

				if (light->mLightType == PhysicsEngine::LightType::Spot || light->mLightType == PhysicsEngine::LightType::Point)
				{
					float shadowNearPlane = light->mShadowNearPlane;
					if (ImGui::SliderFloat("Shadow Near Plane", &shadowNearPlane, 0.1f, 10.0f))
					{
						light->mShadowNearPlane = shadowNearPlane;
					}

					float shadowFarPlane = light->mShadowFarPlane;
					if (ImGui::SliderFloat("Shadow Far Plane", &shadowFarPlane, 10.0f, 250.0f))
					{
						light->mShadowFarPlane = shadowFarPlane;
					}
				}

				const char* shadowMapResolutions[] = { "Low (512x512)", "Medium (1024x1024)", "High (2048x2048)",
												  "Very High (4096x4096)" };
				PhysicsEngine::ShadowMapResolution shadowMapRes = light->getShadowMapResolution();
				int shadowMapResIndex = 0;
				switch (shadowMapRes)
				{
				case PhysicsEngine::ShadowMapResolution::Low512x512:
					shadowMapResIndex = 0;
					break;
				case PhysicsEngine::ShadowMapResolution::Medium1024x1024:
					shadowMapResIndex = 1;
					break;
				case PhysicsEngine::ShadowMapResolution::High2048x2048:
					shadowMapResIndex = 2;
					break;
				case PhysicsEngine::ShadowMapResolution::VeryHigh4096x4096:
					shadowMapResIndex = 3;
					break;
				}
				if (ImGui::Combo("Shadow Map Resolution", &shadowMapResIndex, shadowMapResolutions,
					IM_ARRAYSIZE(shadowMapResolutions)))
				{
					switch (shadowMapResIndex)
					{
					case 0:
						light->setShadowMapResolution(PhysicsEngine::ShadowMapResolution::Low512x512);
						break;
					case 1:
						light->setShadowMapResolution(PhysicsEngine::ShadowMapResolution::Medium1024x1024);
						break;
					case 2:
						light->setShadowMapResolution(PhysicsEngine::ShadowMapResolution::High2048x2048);
						break;
					case 3:
						light->setShadowMapResolution(PhysicsEngine::ShadowMapResolution::VeryHigh4096x4096);
						break;
					}
				}
			}

			bool enabled = light->mEnabled;
			if (ImGui::Checkbox("Enabled?", &enabled))
			{
				light->mEnabled = enabled;
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
				PhysicsEngine::Light* light = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Light>(id);
				clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(light->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::Light>::type);
			}

			ImGui::EndPopup();
		}
	}
}

bool LightDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}