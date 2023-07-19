#ifndef TERRAIN_DRAWER_H__
#define TERRAIN_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

#include <graphics/Framebuffer.h>

namespace PhysicsEditor
{
	class TerrainDrawer
	{
	private:
		PhysicsEngine::Framebuffer* mFBO;
		PhysicsEngine::ShaderProgram* mProgram;

		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		TerrainDrawer();
		~TerrainDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif