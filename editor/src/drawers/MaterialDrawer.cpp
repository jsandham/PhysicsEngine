#include "../../include/drawers/MaterialDrawer.h"

#include "components/Camera.h"
#include "components/Light.h"
#include "components/Transform.h"

#include "core/MaterialUtil.h"
#include "core/Mesh.h"
#include "core/Shader.h"
#include "core/Texture2D.h"

#include "systems/CleanUpSystem.h"
#include "systems/RenderSystem.h"

#include "graphics/Graphics.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{
    mCameraPos = glm::vec3(0.0f, 0.0f, -2.0);
    mModel = glm::mat4(1.0f);
    mView = glm::lookAt(mCameraPos, mCameraPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0, 1.0f, 0.0f));
    mProjection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 10.0f);

    Graphics::createFramebuffer(1000, 1000, &mFBO, &mColor, &mDepth);

    Graphics::createGlobalCameraUniforms(mCameraUniform);
    Graphics::createGlobalLightUniforms(mLightUniform);

    mCameraUniform.mView = mView;
    mCameraUniform.mProjection = mProjection;
    mCameraUniform.mCameraPos = mCameraPos;

    mLightUniform.mIntensity = 1.0f;
    mLightUniform.mShadowNearPlane = 0.1f;
    mLightUniform.mShadowFarPlane = 10.0f;
    mLightUniform.mShadowBias = 0.005f;
    mLightUniform.mShadowRadius = 0.0f;
    mLightUniform.mShadowStrength = 1.0f;
}

MaterialDrawer::~MaterialDrawer()
{
    Graphics::destroyFramebuffer(&mFBO, &mColor, &mDepth);
}

void MaterialDrawer::render(Clipboard &clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Material *material = clipboard.getWorld()->getAssetById<Material>(id);

    Guid currentShaderId = material->getShaderId();

    Shader *ss = clipboard.getWorld()->getAssetById<Shader>(currentShaderId);

    if (ImGui::BeginCombo("Shader", (ss == nullptr ? "" : ss->getName()).c_str(), ImGuiComboFlags_None))
    {
        for (int i = 0; i < clipboard.getWorld()->getNumberOfAssets<Shader>(); i++)
        {
            Shader *s = clipboard.getWorld()->getAssetByIndex<Shader>(i);

            std::string label = s->getName() + "##" + s->getId().toString();

            bool is_selected = (currentShaderId == s->getId());
            if (ImGui::Selectable(label.c_str(), is_selected))
            {
                currentShaderId = s->getId();

                material->setShaderId(currentShaderId);

                material->onShaderChanged(clipboard.getWorld());
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (currentShaderId.isInvalid()) {
        // material has no shader assigned to it
        return;
    }

    // draw material uniforms
    std::vector<ShaderUniform> uniforms = material->getUniforms();
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        // Note: matrices not supported
        switch (uniforms[i].mType)
        {
        case ShaderUniformType::Int:
            UniformDrawer<ShaderUniformType::Int>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::Float:
            UniformDrawer<ShaderUniformType::Float>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::Vec2:
            UniformDrawer<ShaderUniformType::Vec2>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::Vec3:
            UniformDrawer<ShaderUniformType::Vec3>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::Vec4:
            UniformDrawer<ShaderUniformType::Vec4>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::Sampler2D:
            UniformDrawer<ShaderUniformType::Sampler2D>::draw(clipboard, material, &uniforms[i]);
            break;
        case ShaderUniformType::SamplerCube:
            UniformDrawer<ShaderUniformType::SamplerCube>::draw(clipboard, material, &uniforms[i]);
            break;
        }
    }

    ImGui::Separator();

    // Draw material preview child window
    ImGui::Text("Preview");

    Mesh *mesh = clipboard.getWorld()->getAssetById<Mesh>(clipboard.getWorld()->getAssetId("data\\meshes\\sphere.mesh"));
    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(currentShaderId);

    if (mesh == nullptr || shader == nullptr) {
        return;
    }

    Graphics::setGlobalCameraUniforms(mCameraUniform);
    Graphics::setGlobalLightUniforms(mLightUniform);

    int shaderProgram = shader->getProgramFromVariant(static_cast<int64_t>(ShaderMacro::None));

    shader->use(shaderProgram);
    shader->setMat4("model", mModel);

    material->apply(clipboard.getWorld());

    Graphics::bindFramebuffer(mFBO);
    Graphics::setViewport(0, 0, 1000, 1000);
    Graphics::clearFrambufferColor(Color(0.0f, 0.0, 0.0, 1.0f));
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::render(0, (int)mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO());
    Graphics::unbindFramebuffer();

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_None; // ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ?
                               // ImGuiWindowFlags_NoScrollWithMouse : 0);
    ImGui::BeginChild("MaterialPreviewWindow",
                      ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true,
                      window_flags);
    ImGui::Image((void *)(intptr_t)mColor,
                 ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                 ImVec2(0, 0));
    ImGui::EndChild();

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}