#ifndef RENDER_TEXTURE_DRAWER_H__
#define RENDER_TEXTURE_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class RenderTextureDrawer : public InspectorDrawer
    {
    public:
        RenderTextureDrawer();
        ~RenderTextureDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };
} // namespace PhysicsEditor

#endif