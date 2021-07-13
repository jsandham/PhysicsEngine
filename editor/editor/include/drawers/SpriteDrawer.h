#ifndef __SPRITE_DRAWER_H__
#define __SPRITE_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

#include "components/SpriteRenderer.h"
#include "core/Sprite.h"
#include "core/World.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../../include/imgui/imgui_extensions.h"

namespace PhysicsEditor
{
    class SpriteDrawer : public InspectorDrawer
    {
    public:
        SpriteDrawer();
        ~SpriteDrawer();

        void render(Clipboard& clipboard, Guid id);
    };
} // namespace PhysicsEditor

#endif