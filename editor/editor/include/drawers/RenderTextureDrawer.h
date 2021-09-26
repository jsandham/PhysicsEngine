#ifndef __RENDER_TEXTURE_DRAWER_H__
#define __RENDER_TEXTURE_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class RenderTextureDrawer : public InspectorDrawer
    {
    public:
        RenderTextureDrawer();
        ~RenderTextureDrawer();

        virtual void render(Clipboard& clipboard, Guid id) override;
    };
} // namespace PhysicsEditor

#endif