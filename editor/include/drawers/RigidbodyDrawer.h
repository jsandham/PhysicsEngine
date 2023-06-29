#ifndef RIGIDBODY_DRAWER_H__
#define RIGIDBODY_DRAWER_H__

#include <imgui.h>
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class RigidbodyDrawer
{
    private:
        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        RigidbodyDrawer();
        ~RigidbodyDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
};
} // namespace PhysicsEditor

#endif