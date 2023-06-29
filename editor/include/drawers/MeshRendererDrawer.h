#ifndef MESHRENDERER_DRAWER_H__
#define MESHRENDERER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class MeshRendererDrawer
{
    private:
        ImVec2 mContentMin;
        ImVec2 mContentMax;
  
    public:
        MeshRendererDrawer();
        ~MeshRendererDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
};
} // namespace PhysicsEditor

#endif