#ifndef SPRITE_DRAWER_H__
#define SPRITE_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class SpriteDrawer : public InspectorDrawer
    {
    public:
        SpriteDrawer();
        ~SpriteDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };
} // namespace PhysicsEditor

#endif