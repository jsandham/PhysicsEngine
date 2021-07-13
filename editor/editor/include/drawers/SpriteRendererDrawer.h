#ifndef __SPRITERENDERER_DRAWER_H__
#define __SPRITERENDERER_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class SpriteRendererDrawer : public InspectorDrawer
    {
    public:
        SpriteRendererDrawer();
        ~SpriteRendererDrawer();

        void render(Clipboard& clipboard, Guid id);
    };
} // namespace PhysicsEditor

#endif