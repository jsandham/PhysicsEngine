#include "../../include/drawers/LightDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/Light.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

LightDrawer::LightDrawer()
{
}

LightDrawer::~LightDrawer()
{
}

void LightDrawer::render(Clipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Light *light = clipboard.getWorld()->getComponentById<Light>(id);

        ImGui::Text(("EntityId: " + light->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        glm::vec4 color = light->mColor;

        if (ImGui::InputFloat4("Color", glm::value_ptr(color)))
        {
            Undo::recordComponent(light);

            light->mColor = color;
        }

        float intensity = light->mIntensity;
        float spotAngle = light->mSpotAngle;
        float innerSpotAngle = light->mInnerSpotAngle;
        float shadowNearPlane = light->mShadowNearPlane;
        float shadowFarPlane = light->mShadowFarPlane;
        float shadowAngle = light->mShadowAngle;
        float shadowRadius = light->mShadowRadius;
        float shadowStrength = light->mShadowStrength;

        if (ImGui::InputFloat("Intensity", &intensity))
        {
            Undo::recordComponent(light);

            light->mIntensity = intensity;
        }
        if (ImGui::InputFloat("Spot Angle", &spotAngle))
        {
            Undo::recordComponent(light);

            light->mSpotAngle = spotAngle;
        }
        if (ImGui::InputFloat("Inner Spot Angle", &innerSpotAngle))
        {
            Undo::recordComponent(light);

            light->mInnerSpotAngle = innerSpotAngle;
        }
        if (ImGui::InputFloat("Shadow Near Plane", &shadowNearPlane))
        {
            Undo::recordComponent(light);

            light->mShadowNearPlane = shadowNearPlane;
        }
        if (ImGui::InputFloat("Shadow Far Plane", &shadowFarPlane))
        {
            Undo::recordComponent(light);

            light->mShadowFarPlane = shadowFarPlane;
        }
        if (ImGui::InputFloat("Shadow Angle", &shadowAngle))
        {
            Undo::recordComponent(light);

            light->mShadowAngle = shadowAngle;
        }
        if (ImGui::InputFloat("Shadow Radius", &shadowRadius))
        {
            Undo::recordComponent(light);

            light->mShadowRadius = shadowRadius;
        }
        if (ImGui::InputFloat("Shadow Strength", &shadowStrength))
        {
            Undo::recordComponent(light);

            light->mShadowStrength = shadowStrength;
        }

        const char *lightTypes[] = {"Directional", "Spot", "Point"};
        int lightTypeIndex = static_cast<int>(light->mLightType);
        if (ImGui::Combo("##LightType", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes)))
        {
            Undo::recordComponent(light);

            light->mLightType = static_cast<LightType>(lightTypeIndex);
        }

        const char *shadowTypes[] = {"Hard", "Soft"};
        int shadowTypeIndex = static_cast<int>(light->mShadowType);
        if (ImGui::Combo("##ShadowType", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes)))
        {
            Undo::recordComponent(light);

            light->mShadowType = static_cast<ShadowType>(shadowTypeIndex);
        }

        const char *shadowMapResolutions[] = {"Low (512x512)", "Medium (1024x1024)", "High (2048x2048)",
                                              "Very High (4096x4096)"};
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
        if (ImGui::Combo("##ShadowMapResolution", &shadowMapResIndex, shadowMapResolutions,
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

        ImGui::TreePop();
    }
}