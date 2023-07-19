#ifndef CUBEMAP_DRAWER_H__
#define CUBEMAP_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class CubemapDrawer
	{
	private:
		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		CubemapDrawer();
		~CubemapDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		void drawCubemapFaceTexture(Clipboard& clipboard, PhysicsEngine::CubemapFace face, PhysicsEngine::Cubemap* cubemap, PhysicsEngine::Texture2D* texture);
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif