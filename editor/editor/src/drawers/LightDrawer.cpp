#include "../../include/drawers/LightDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/Light.h"

#include "imgui.h"

using namespace PhysicsEditor;

LightDrawer::LightDrawer()
{
}

LightDrawer::~LightDrawer()
{
}

void LightDrawer::render(Clipboard &clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Light *light = clipboard.getWorld()->getComponentById<Light>(id);

        if (light != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            const char* lightTypes[] = { "Directional", "Spot", "Point" };
            int lightTypeIndex = static_cast<int>(light->mLightType);
            if (ImGui::Combo("Light Type", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes)))
            {
                light->mLightType = static_cast<LightType>(lightTypeIndex);
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

            if (light->mLightType == LightType::Spot)
            {
                float spotAngleRad = glm::radians(light->mSpotAngle);
                float innerSpotAngleRad = glm::radians(light->mInnerSpotAngle);

                if (ImGui::SliderAngle("Spot Angle", &spotAngleRad, 0.0f, 179.0f))
                {
                    light->mSpotAngle = glm::degrees(spotAngleRad);
                }
                if (ImGui::SliderAngle("Inner Spot Angle", &innerSpotAngleRad, 0.0f, 179.0f))
                {
                    light->mInnerSpotAngle = glm::degrees(innerSpotAngleRad);
                }
            }

            const char* shadowTypes[] = { "Hard Shadows", "Soft Shadows", "No Shadows" };
            int shadowTypeIndex = static_cast<int>(light->mShadowType);
            if (ImGui::Combo("Shadow Type", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes)))
            {
                light->mShadowType = static_cast<ShadowType>(shadowTypeIndex);
            }

            if (light->mShadowType != ShadowType::None)
            {
                float shadowStrength = light->mShadowStrength;
                if (ImGui::SliderFloat("Shadow Strength", &shadowStrength, 0.0f, 1.0f))
                {
                    light->mShadowStrength = shadowStrength;
                }

                const char* shadowMapResolutions[] = { "Low (512x512)", "Medium (1024x1024)", "High (2048x2048)",
                                                  "Very High (4096x4096)" };
                ShadowMapResolution shadowMapRes = light->getShadowMapResolution();
                int shadowMapResIndex = 0;
                switch (shadowMapRes)
                {
                case ShadowMapResolution::Low512x512:
                    shadowMapResIndex = 0;
                    break;
                case ShadowMapResolution::Medium1024x1024:
                    shadowMapResIndex = 1;
                    break;
                case ShadowMapResolution::High2048x2048:
                    shadowMapResIndex = 2;
                    break;
                case ShadowMapResolution::VeryHigh4096x4096:
                    shadowMapResIndex = 3;
                    break;
                }
                if (ImGui::Combo("Shadow Map Resolution", &shadowMapResIndex, shadowMapResolutions,
                    IM_ARRAYSIZE(shadowMapResolutions)))
                {
                    switch (shadowMapResIndex)
                    {
                    case 0:
                        light->setShadowMapResolution(ShadowMapResolution::Low512x512);
                        break;
                    case 1:
                        light->setShadowMapResolution(ShadowMapResolution::Medium1024x1024);
                        break;
                    case 2:
                        light->setShadowMapResolution(ShadowMapResolution::High2048x2048);
                        break;
                    case 3:
                        light->setShadowMapResolution(ShadowMapResolution::VeryHigh4096x4096);
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
}