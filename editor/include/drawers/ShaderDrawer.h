#ifndef SHADER_DRAWER_H__
#define SHADER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class ShaderDrawer
{
    private:
        PhysicsEngine::Guid mShaderId;

        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        ShaderDrawer();
        ~ShaderDrawer();

    void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
};
} // namespace PhysicsEditor

#endif