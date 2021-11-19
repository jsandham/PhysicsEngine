#ifndef __SPRITERENDERER_DRAWER_H__
#define __SPRITERENDERER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class SpriteRendererDrawer : public InspectorDrawer
    {
    public:
        SpriteRendererDrawer();
        ~SpriteRendererDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };
} // namespace PhysicsEditor

#endif