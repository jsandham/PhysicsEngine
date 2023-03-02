#include "../../include/drawers/MaterialDrawer.h"
#include "../../include/ProjectDatabase.h"

#define GLM_FORCE_RADIANS

#include "components/Camera.h"
#include "components/Light.h"
#include "components/Transform.h"

#include "core/Mesh.h"
#include "core/Shader.h"
#include "core/Texture2D.h"

#include "graphics/Renderer.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{
    mCameraPos = glm::vec3(0.0f, 0.0f, -4.0);
    mModel = glm::mat4(1.0f);
    mView = glm::lookAt(mCameraPos, mCameraPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0, 1.0f, 0.0f));
    mProjection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 10.0f);
    mViewProjection = mProjection * mView;

    mFBO = Framebuffer::create(1000, 1000);

    mCameraUniform = RendererUniforms::getCameraUniform();
    mLightUniform = RendererUniforms::getLightUniform();

    mDrawRequired = true;
}

MaterialDrawer::~MaterialDrawer()
{
    delete mFBO;
}

void MaterialDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Material *material = clipboard.getWorld()->getAssetByGuid<Material>(id);

    if (material != nullptr)
    {
        ImGui::Text(("Material id: " + material->getGuid().toString()).c_str());
        ImGui::Text(("Shader id: " + material->getShaderId().toString()).c_str());

        Guid currentShaderId = material->getShaderId();

        Shader* ss = clipboard.getWorld()->getAssetByGuid<Shader>(currentShaderId);

        if (ImGui::BeginCombo("Shader", (ss == nullptr ? "" : ss->getName()).c_str(), ImGuiComboFlags_None))
        {
            for (int i = 0; i < clipboard.getWorld()->getNumberOfAssets<Shader>(); i++)
            {
                Shader* s = clipboard.getWorld()->getAssetByIndex<Shader>(i);

                std::string label = s->getName() + "##" + s->getGuid().toString();

                bool is_selected = (currentShaderId == s->getGuid());
                if (ImGui::Selectable(label.c_str(), is_selected))
                {
                    currentShaderId = s->getGuid();

                    material->setShaderId(currentShaderId);
                    material->onShaderChanged();
                    clipboard.mModifiedAssets.insert(material->getGuid());

                    mDrawRequired = true;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        Shader* shader = clipboard.getWorld()->getAssetByGuid<Shader>(currentShaderId);

        if (shader == nullptr) 
        {
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
                this->drawIntUniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Float:
                this->drawFloatUniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Color:
                this->drawColorUniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Vec2:
                this->drawVec2Uniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Vec3:
                this->drawVec3Uniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Vec4:
                this->drawVec4Uniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::Sampler2D:
                this->drawTexture2DUniform(clipboard, material, &uniforms[i]);
                break;
            case ShaderUniformType::SamplerCube:
                this->drawCubemapUniform(clipboard, material, &uniforms[i]);
                break;
            }
        }

        if (ImGui::Checkbox("Enable Instancing", &material->mEnableInstancing))
        {
            mDrawRequired = true;
        }

        ImGui::Separator();

        // Draw material preview child window
        ImGui::Text("Preview");

        // A change to the material was made so re-draw required
        if(mDrawRequired)
        {
            Mesh* mesh = clipboard.getWorld()->getPrimtiveMesh(PhysicsEngine::PrimitiveType::Sphere);
            if (mesh != nullptr)
            {
                mCameraUniform->setView(mView);
                mCameraUniform->setProjection(mProjection);
                mCameraUniform->setViewProjection(mViewProjection);
                mCameraUniform->setCameraPos(mCameraPos);

                mLightUniform->setLightIntensity(1.0f);
                mLightUniform->setShadowNearPlane(0.1f);
                mLightUniform->setShadowFarPlane(10.0f);
                mLightUniform->setShadowBias(0.005f);
                mLightUniform->setShadowRadius(0.0f);
                mLightUniform->setShadowStrength(1.0f);

                mCameraUniform->copyToUniformsToDevice();
                mLightUniform->copyToUniformsToDevice();

                int64_t variant = 0;
                variant |= static_cast<int64_t>(ShaderMacro::Directional);
                variant |= static_cast<int64_t>(ShaderMacro::HardShadows);

                shader->bind(shader->getProgramFromVariant(variant) == nullptr ? 0 : variant);
                shader->setMat4("model", mModel);

                material->apply();

                mFBO->bind();
                mFBO->setViewport(0, 0, 1000, 1000);
                mFBO->clearColor(Color(0.15f, 0.15f, 0.15f, 1.0f));
                mFBO->clearDepth(1.0f);
                /*mesh->getNativeGraphicsHandle()->draw(0, mesh->getVertices().size() / 3);*/
                mesh->getNativeGraphicsHandle()->drawIndexed(0, mesh->getIndices().size());
                mFBO->unbind();
            }

            mDrawRequired = false;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
        ImGui::BeginChild("MaterialPreviewWindow",
            ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true,
            window_flags);
        ImGui::Image((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle())),
            ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
            ImVec2(0, 0));
        ImGui::EndChild();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}

void MaterialDrawer::drawIntUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    int temp = material->getInt(uniform->mName);

    if (ImGui::InputInt(uniform->getShortName().c_str(), &temp))
    {
        material->setInt(uniform->mName, temp);
        clipboard.mModifiedAssets.insert(material->getGuid());
        mDrawRequired = true;
    }
}

void MaterialDrawer::drawFloatUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    float temp = material->getFloat(uniform->mName);

    if (ImGui::InputFloat(uniform->getShortName().c_str(), &temp))
    {
        material->setFloat(uniform->mName, temp);
        clipboard.mModifiedAssets.insert(material->getGuid());
        mDrawRequired = true;
    }
}

void MaterialDrawer::drawColorUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    Color temp = material->getColor(uniform->mName);

    if (ImGui::ColorEdit4(uniform->getShortName().c_str(), reinterpret_cast<float*>(&temp.mR)))
    {
        material->setColor(uniform->mName, temp);
        clipboard.mModifiedAssets.insert(material->getGuid());
        mDrawRequired = true;
    }
}

void MaterialDrawer::drawVec2Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec2 temp = material->getVec2(uniform->mName);

    if (ImGui::InputFloat2(uniform->getShortName().c_str(), &temp[0]))
    {
        material->setVec2(uniform->mName, temp);
        clipboard.mModifiedAssets.insert(material->getGuid());
        mDrawRequired = true;
    }
}

void MaterialDrawer::drawVec3Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec3 temp = material->getVec3(uniform->mName);

    if (uniform->mName.find("color") != std::string::npos ||
        uniform->mName.find("colour") != std::string::npos)
    {
        if (ImGui::ColorEdit3(uniform->getShortName().c_str(), reinterpret_cast<float*>(&temp.x)))
        {
            material->setVec3(uniform->mName, temp);
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
    }
    else
    {
        if (ImGui::InputFloat3(uniform->getShortName().c_str(), &temp[0]))
        {
            material->setVec3(uniform->mName, temp);
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
    }
}

void MaterialDrawer::drawVec4Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    glm::vec4 temp = material->getVec4(uniform->mName);

    if (uniform->mName.find("color") != std::string::npos ||
        uniform->mName.find("colour") != std::string::npos)
    {
        if (ImGui::ColorEdit4(uniform->getShortName().c_str(), reinterpret_cast<float*>(&temp.x)))
        {
            material->setVec4(uniform->mName, temp);
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
    }
    else
    {
        if (ImGui::InputFloat4(uniform->getShortName().c_str(), &temp[0]))
        {
            material->setVec4(uniform->mName, temp);
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
    }
}

void MaterialDrawer::drawTexture2DUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    Texture2D* texture = clipboard.getWorld()->getAssetByGuid<Texture2D>(material->getTexture(uniform->mName));

    if (ImGui::ImageButton((void*)(intptr_t)(texture == nullptr ? 0 : *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle())),
        ImVec2(80, 80),
        ImVec2(1, 1),
        ImVec2(0, 0),
        0,
        ImVec4(1, 1, 1, 1),
        ImVec4(1, 1, 1, 0.5)))
    {
        if(texture != nullptr)
        {
            clipboard.setSelectedItem(InteractionType::Texture2D, texture->getGuid());
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
    }

    if (ImGui::BeginDragDropTarget())
    {
        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("TEXTURE2D_PATH");
        if (payload != nullptr)
        {
            const char* data = static_cast<const char*>(payload->Data);

            material->setTexture(uniform->mName, ProjectDatabase::getGuid(data));
            material->onTextureChanged();
            clipboard.mModifiedAssets.insert(material->getGuid());
            mDrawRequired = true;
        }
        ImGui::EndDragDropTarget();
    }
}

void MaterialDrawer::drawCubemapUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform)
{
    
}
