#ifndef TERRAIN_DRAWER_H__
#define TERRAIN_DRAWER_H__

#include "InspectorDrawer.h"
#include <graphics/Framebuffer.h>

namespace PhysicsEditor
{
    class TerrainDrawer : public InspectorDrawer
    {
    private:
        Framebuffer* mFBO;
        ShaderProgram* mProgram;

    public:
        TerrainDrawer();
        ~TerrainDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };
} // namespace PhysicsEditor

#endif