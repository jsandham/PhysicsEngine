#ifndef SCENE_DRAWER_H__
#define SCENE_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
    class SceneDrawer
    {
    private:
        ImVec2 mContentMin;
        ImVec2 mContentMax;

    public:
        SceneDrawer();
        ~SceneDrawer();

        void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

    private:
        bool isHovered() const;
    };

} // namespace PhysicsEditor

#endif