#ifndef TEXTURE2D_DRAWER_H__
#define TEXTURE2D_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"
#include <graphics/Framebuffer.h>
#include <graphics/ShaderProgram.h>
#include <graphics/TextureHandle.h>
#include <graphics/RendererMeshes.h>

namespace PhysicsEditor
{
	class Texture2DDrawer
	{
	private:
		PhysicsEngine::Framebuffer* mFBO;

		PhysicsEngine::ShaderProgram* mProgramR;
		PhysicsEngine::ShaderProgram* mProgramG;
		PhysicsEngine::ShaderProgram* mProgramB;
		PhysicsEngine::ShaderProgram* mProgramA;

		PhysicsEngine::ScreenQuad* mScreenQuad;

		PhysicsEngine::Guid mCurrentTexId;
		void* mDrawTex;

		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		Texture2DDrawer();
		~Texture2DDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif