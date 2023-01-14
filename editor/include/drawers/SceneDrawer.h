#ifndef SCENE_DRAWER_H__
#define SCENE_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class SceneDrawer : public InspectorDrawer
    {
    public:
        SceneDrawer();
        ~SceneDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };

} // namespace PhysicsEditor

#endif