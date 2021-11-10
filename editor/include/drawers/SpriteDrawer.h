#ifndef __SPRITE_DRAWER_H__
#define __SPRITE_DRAWER_H__

#include "InspectorDrawer.h"

#include "components/SpriteRenderer.h"
#include "core/Sprite.h"
#include "core/World.h"

#include "imgui.h"

#include "../../include/imgui/imgui_extensions.h"

namespace PhysicsEditor
{
    class SpriteDrawer : public InspectorDrawer
    {
    public:
        SpriteDrawer();
        ~SpriteDrawer();

        virtual void render(Clipboard& clipboard, Guid id) override;
    };
} // namespace PhysicsEditor

#endif