#ifndef LIGHT_DRAWER_H__
#define LIGHT_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class LightDrawer
{
    private:
        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        LightDrawer();
        ~LightDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
};
} // namespace PhysicsEditor

#endif