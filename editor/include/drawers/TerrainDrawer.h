#ifndef __TERRAIN_DRAWER_H__
#define __TERRAIN_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
    class TerrainDrawer : public InspectorDrawer
    {
    public:
        TerrainDrawer();
        ~TerrainDrawer();

        virtual void render(Clipboard& clipboard, const Guid& id) override;
    };
} // namespace PhysicsEditor

#endif