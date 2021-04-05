#include "../../include/drawers/MaterialDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/Camera.h"
#include "components/Light.h"
#include "components/Transform.h"

#include "core/InternalMeshes.h"
#include "core/MaterialUtil.h"
#include "core/Mesh.h"
#include "core/Shader.h"
#include "core/Texture2D.h"

#include "systems/CleanUpSystem.h"
#include "systems/RenderSystem.h"

#include "graphics/Graphics.h"

#include "../../include/EditorCameraSystem.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{
    cameraPos = glm::vec3(0.0f, 0.0f, -2.0);
    model = glm::mat4(1.0f);
    view = glm::lookAt(cameraPos, cameraPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0, 1.0f, 0.0f));
    projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 10.0f);

    Graphics::createFramebuffer(1000, 1000, &mFBO, &mColor, &mDepth);

    Graphics::createGlobalCameraUniforms(cameraUniform);
    Graphics::createGlobalLightUniforms(lightUniform);

    cameraUniform.mView = view;
    cameraUniform.mProjection = projection;
    cameraUniform.mCameraPos = cameraPos;

    lightUniform.mIntensity = 1.0f;
    lightUniform.mShadowNearPlane = 0.1f;
    lightUniform.mShadowFarPlane = 10.0f;
    lightUniform.mShadowAngle = 0.0f;
    lightUniform.mShadowRadius = 0.0f;
    lightUniform.mShadowStrength = 1.0f;
}

MaterialDrawer::~MaterialDrawer()
{
    Graphics::destroyFramebuffer(&mFBO, &mColor, &mDepth);
}

void MaterialDrawer::render(Clipboard &clipboard, Guid id)
{
    Material *material = clipboard.getWorld()->getAssetById<Material>(id);

    if (material == nullptr)
    {
        std::string test = id.toString();
    }

    Guid currentShaderId = material->getShaderId();

    Shader *ss = clipboard.getWorld()->getAssetById<Shader>(currentShaderId);

    if (ImGui::BeginCombo("Shader", ss->getName().c_str(), ImGuiComboFlags_None))
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

    // draw material uniforms
    std::vector<ShaderUniform> uniforms = material->getUniforms();
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        // only expose uniforms that exist in a Material uniform struct in the shader
        if (std::strcmp(uniforms[i].mBlockName, "material") != 0)
        {
            continue;
        }

        // Note: matrices not supported
        switch (uniforms[i].mType)
        {
        case GL_INT:
            UniformDrawer<GL_INT>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_FLOAT:
            UniformDrawer<GL_FLOAT>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_FLOAT_VEC2:
            UniformDrawer<GL_FLOAT_VEC2>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_FLOAT_VEC3:
            UniformDrawer<GL_FLOAT_VEC3>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_FLOAT_VEC4:
            UniformDrawer<GL_FLOAT_VEC4>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_SAMPLER_2D:
            UniformDrawer<GL_SAMPLER_2D>::draw(clipboard, material, &uniforms[i]);
            break;
        case GL_SAMPLER_CUBE:
            UniformDrawer<GL_SAMPLER_CUBE>::draw(clipboard, material, &uniforms[i]);
            break;
        }
    }

    ImGui::Separator();

    // Draw material preview child window
    ImGui::Text("Preview");

    Mesh *mesh = clipboard.getWorld()->getAssetById<Mesh>(clipboard.getWorld()->getSphereMesh());
    assert(mesh != NULL);

    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(currentShaderId);
    assert(shader != NULL);

    Graphics::setGlobalCameraUniforms(cameraUniform);
    Graphics::setGlobalLightUniforms(lightUniform);

    int shaderProgram = shader->getProgramFromVariant(ShaderVariant::None);

    shader->use(shaderProgram);
    shader->setMat4("model", model);

    material->apply(clipboard.getWorld());

    Graphics::bindFramebuffer(mFBO);
    Graphics::setViewport(0, 0, 1000, 1000);
    Graphics::clearFrambufferColor(Color(0.0f, 0.0, 0.0, 1.0f));
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::render(0, mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO());
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
}