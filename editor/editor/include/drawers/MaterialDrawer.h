#ifndef __MATERIAL_DRAWER_H__
#define __MATERIAL_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

#include "components/MeshRenderer.h"
#include "core/Material.h"
#include "core/World.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../../include/imgui/imgui_extensions.h"

namespace PhysicsEditor
{
class MaterialDrawer : public InspectorDrawer
{
  private:
    GLuint mFBO;
    GLuint mColor;
    GLuint mDepth;

    CameraUniform mCameraUniform;
    LightUniform mLightUniform;

    glm::vec3 mCameraPos;
    glm::mat4 mModel;
    glm::mat4 mView;
    glm::mat4 mProjection;

  public:
    MaterialDrawer();
    ~MaterialDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};

template <GLenum T> struct UniformDrawer
{
    static void draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform);
};

template <GLenum T>
inline void UniformDrawer<T>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
}

template <>
inline void UniformDrawer<GL_INT>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    int temp = material->getInt(uniform->mName);

    /*if (ImGui::InputInt(uniform->mShortName, &temp))
    {
        material->setInt(uniform->mName, temp);
    }*/
    if (ImGui::InputInt(uniform->mName.c_str(), &temp))
    {
        material->setInt(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<GL_FLOAT>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    float temp = material->getFloat(uniform->mName);

    /*if (ImGui::InputFloat(uniform->mShortName, &temp))
    {
        material->setFloat(uniform->mName, temp);
    }*/
    if (ImGui::InputFloat(uniform->mName.c_str(), &temp))
    {
        material->setFloat(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<GL_FLOAT_VEC2>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    glm::vec2 temp = material->getVec2(uniform->mName);

    /*if (ImGui::InputFloat2(uniform->mShortName, &temp[0]))
    {
        material->setVec2(uniform->mName, temp);
    }*/
    if (ImGui::InputFloat2(uniform->mName.c_str(), &temp[0]))
    {
        material->setVec2(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<GL_FLOAT_VEC3>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    glm::vec3 temp = material->getVec3(uniform->mName);

    /*if (ImGui::InputFloat3(uniform->mShortName, &temp[0]))
    {
        material->setVec3(uniform->mName, temp);
    }*/
    if (ImGui::InputFloat3(uniform->mName.c_str(), &temp[0]))
    {
        material->setVec3(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<GL_FLOAT_VEC4>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    glm::vec4 temp = material->getVec4(uniform->mName);

    /*if (ImGui::InputFloat4(uniform->mShortName, &temp[0]))
    {
        material->setVec4(uniform->mName, temp);
    }*/
    if (ImGui::InputFloat4(uniform->mName.c_str(), &temp[0]))
    {
        material->setVec4(uniform->mName, temp);
    }
}

template <>
inline void UniformDrawer<GL_SAMPLER_2D>::draw(Clipboard &clipboard, Material *material, ShaderUniform *uniform)
{
    Texture2D *texture = clipboard.getWorld()->getAssetById<Texture2D>(material->getTexture(uniform->mName));

    bool releaseTriggered = false;
    bool clearClicked = false;
    /*bool isClicked = ImGui::ImageSlot(uniform->mShortName, texture == nullptr ? 0 : texture->getNativeGraphics(), &releaseTriggered, &clearClicked);*/
    bool isClicked = ImGui::ImageSlot(uniform->mName, texture == nullptr ? 0 : texture->getNativeGraphics(), &releaseTriggered, &clearClicked);
    
    if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Texture2D)
    {
        material->setTexture(uniform->mName, clipboard.getDraggedId());
        clipboard.clearDraggedItem();
    }
    
    if (clearClicked)
    {
        material->setTexture(uniform->mName, Guid::INVALID);
    }

    if (isClicked)
    {
        if (material->getTexture(uniform->mName).isValid())
        {
            clipboard.setSelectedItem(InteractionType::Texture2D, material->getTexture(uniform->mName));
        }
    }
}
} // namespace PhysicsEditor

#endif