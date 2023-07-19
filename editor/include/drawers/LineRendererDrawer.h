#ifndef LINERENDERER_DRAWER_H__
#define LINERENDERER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class LineRendererDrawer
	{
	private:
		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		LineRendererDrawer();
		~LineRendererDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif