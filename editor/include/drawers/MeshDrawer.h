#ifndef MESH_DRAWER_H__
#define MESH_DRAWER_H__

#include <imgui.h>

#include <graphics/Framebuffer.h>
#include <graphics/RendererUniforms.h>
#include <graphics/VertexBuffer.h>
#include <graphics/MeshHandle.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class MeshDrawer
	{
	private:
		PhysicsEngine::Framebuffer* mFBO;

		PhysicsEngine::CameraUniform* mCameraUniform;

		glm::mat4 mModel;

		float mMouseX;
		float mMouseY;

		int mActiveDrawModeIndex;
		bool mWireframeOn;
		bool mResetModelMatrix;

		ImVec2 mContentMin;
		ImVec2 mContentMax;

		PhysicsEngine::VertexBuffer* mVertexBuffer;
		PhysicsEngine::VertexBuffer* mNormalBuffer;
		PhysicsEngine::MeshHandle* mMeshHandle;


	public:
		MeshDrawer();
		~MeshDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif