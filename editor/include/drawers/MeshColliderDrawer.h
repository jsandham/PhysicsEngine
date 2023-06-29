#ifndef MESHCOLLIDER_DRAWER_H__
#define MESHCOLLIDER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class MeshColliderDrawer
{
    private:
        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        MeshColliderDrawer();
        ~MeshColliderDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
};
} // namespace PhysicsEditor

#endif