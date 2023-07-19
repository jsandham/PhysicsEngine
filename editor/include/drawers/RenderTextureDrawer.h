#ifndef RENDER_TEXTURE_DRAWER_H__
#define RENDER_TEXTURE_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class RenderTextureDrawer
	{
	private:
		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		RenderTextureDrawer();
		~RenderTextureDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif