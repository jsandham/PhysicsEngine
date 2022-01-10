#include "../../include/drawers/MaterialDrawer.h"

#define GLM_FORCE_RADIANS

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

#include "Windows.h"

using namespace PhysicsEditor;

template <ShaderUniformType T> struct UniformDrawer
{
    static void draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
};

template <ShaderUniformType T>
inline void UniformDrawer<T>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
}

template <>
inline void UniformDrawer<ShaderUniformType::Int>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    int temp = material->getInt(uniform->mName);

    if (ImGui::InputInt(uniform->mName.c_str(), &temp))
    {
        material->setInt(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Float>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    float temp = material->getFloat(uniform->mName);

    if (ImGui::InputFloat(uniform->mName.c_str(), &temp))
    {
        material->setFloat(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Color>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    Color temp = material->getColor(uniform->mName);

    if (ImGui::ColorEdit4(uniform->mName.c_str(), reinterpret_cast<float*>(&temp.mR)))
    {
        material->setColor(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Vec2>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec2 temp = material->getVec2(uniform->mName);

    if (ImGui::InputFloat2(uniform->mName.c_str(), &temp[0]))
    {
        material->setVec2(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Vec3>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec3 temp = material->getVec3(uniform->mName);

    if (uniform->mName.find("color") != std::string::npos ||
        uniform->mName.find("colour") != std::string::npos)
    {
        if (ImGui::ColorEdit3(uniform->mName.c_str(), reinterpret_cast<float*>(&temp.x)))
        {
            material->setVec3(uniform->mName, temp);
        }
    }
    else
    {
        if (ImGui::InputFloat3(uniform->mName.c_str(), &temp[0]))
        {
            material->setVec3(uniform->mName, temp);
        }
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Vec4>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec4 temp = material->getVec4(uniform->mName);

    if (uniform->mName.find("color") != std::string::npos ||
        uniform->mName.find("colour") != std::string::npos)
    {
        if (ImGui::ColorEdit4(uniform->mName.c_str(), reinterpret_cast<float*>(&temp.x)))
        {
            material->setVec4(uniform->mName, temp);
        }
    }
    else
    {
        if (ImGui::InputFloat4(uniform->mName.c_str(), &temp[0]))
        {
            material->setVec4(uniform->mName, temp);
        }
    }
}

template <>
inline void UniformDrawer<ShaderUniformType::Sampler2D>::draw(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    Texture2D* texture = clipboard.getWorld()->getAssetById<Texture2D>(material->getTexture(uniform->mName));

    bool releaseTriggered = false;
    bool clearClicked = false;
    bool isClicked = ImGui::ImageSlot(uniform->mName, texture == nullptr ? 0 : texture->getNativeGraphics(), &releaseTriggered, &clearClicked);

    if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Texture2D)
    {
        material->setTexture(uniform->mName, clipboard.getDraggedId());
        material->onTextureChanged();
        
        clipboard.clearDraggedItem();
    }

    if (clearClicked)
    {
        material->setTexture(uniform->mName, Guid::INVALID);
        material->onTextureChanged();
    }

    if (isClicked)
    {
        if (material->getTexture(uniform->mName).isValid())
        {
            clipboard.setSelectedItem(InteractionType::Texture2D, material->getTexture(uniform->mName));
        }
    }
}

MaterialDrawer::MaterialDrawer()
{
    mCameraPos = glm::vec3(0.0f, 0.0f, -4.0);
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

void MaterialDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Material *material = clipboard.getWorld()->getAssetById<Material>(id);

    ImGui::Text(("Material id: " + material->getId().toString()).c_str());
    ImGui::Text(("Shader id: " + material->getShaderId().toString()).c_str());

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

                material->onShaderChanged();
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
        case ShaderUniformType::Color:
            UniformDrawer<ShaderUniformType::Color>::draw(clipboard, material, &uniforms[i]);
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

    if (ImGui::Checkbox("Enable Instancing", &material->mEnableInstancing))
    {

    }

    ImGui::Separator();

    // Draw material preview child window
    ImGui::Text("Preview");

    Mesh* mesh = clipboard.getWorld()->getPrimtiveMesh(PhysicsEngine::PrimitiveType::Sphere);
    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(currentShaderId);

    if (mesh == nullptr || shader == nullptr) {
        return;
    }

    Graphics::setGlobalCameraUniforms(mCameraUniform);
    Graphics::setGlobalLightUniforms(mLightUniform);

    int64_t variant = 0;
    variant |= static_cast<int64_t>(ShaderMacro::Directional);
    variant |= static_cast<int64_t>(ShaderMacro::HardShadows);

    int shaderProgram = shader->getProgramFromVariant(variant);
    if (shaderProgram == -1)
    {
        // If we dont have the directional light + shadow variant, revert to default variant
        shaderProgram = shader->getProgramFromVariant(0);
    }

    shader->use(shaderProgram);
    shader->setMat4("model", mModel);

    material->apply();

    Graphics::bindFramebuffer(mFBO);
    Graphics::setViewport(0, 0, 1000, 1000);
    Graphics::clearFrambufferColor(Color(0.15f, 0.15f, 0.15f, 1.0f));
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::render(0, (int)mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO());
    Graphics::unbindFramebuffer();

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
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