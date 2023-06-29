#ifndef MATERIAL_DRAWER_H__
#define MATERIAL_DRAWER_H__

#include <imgui.h>

#include "core/Material.h"
#include <graphics/Framebuffer.h>
#include <graphics/RendererUniforms.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class MaterialDrawer
{
    private:
        PhysicsEngine::Framebuffer* mFBO;

        PhysicsEngine::CameraUniform* mCameraUniform;
        PhysicsEngine::LightUniform* mLightUniform;

        glm::vec3 mCameraPos;
        glm::mat4 mModel;
        glm::mat4 mView;
        glm::mat4 mProjection;
        glm::mat4 mViewProjection;

        bool mDrawRequired;

        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        MaterialDrawer();
        ~MaterialDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        void drawIntUniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawFloatUniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawColorUniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawVec2Uniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawVec3Uniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawVec4Uniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawTexture2DUniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);
        void drawCubemapUniform(Clipboard& clipboard, PhysicsEngine::Material* material, PhysicsEngine::ShaderUniform* uniform);

        bool isHovered() const;
};

} // namespace PhysicsEditor

#endif