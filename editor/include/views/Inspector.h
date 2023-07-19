#ifndef INSPECTOR_H__
#define INSPECTOR_H__

#include <vector>

#include "../EditorClipboard.h"

#include "../drawers/SceneDrawer.h"
#include "../drawers/CubemapDrawer.h"
#include "../drawers/MaterialDrawer.h"
#include "../drawers/MeshDrawer.h"
#include "../drawers/ShaderDrawer.h"
#include "../drawers/Texture2DDrawer.h"
#include "../drawers/RenderTextureDrawer.h"

#include "../../include/drawers/BoxColliderDrawer.h"
#include "../../include/drawers/CameraDrawer.h"
#include "../../include/drawers/CapsuleColliderDrawer.h"
#include "../../include/drawers/LightDrawer.h"
#include "../../include/drawers/LineRendererDrawer.h"
#include "../../include/drawers/MeshColliderDrawer.h"
#include "../../include/drawers/MeshRendererDrawer.h"
#include "../../include/drawers/RigidbodyDrawer.h"
#include "../../include/drawers/SphereColliderDrawer.h"
#include "../../include/drawers/TransformDrawer.h"
#include "../../include/drawers/TerrainDrawer.h"

namespace PhysicsEditor
{
	class Inspector
	{
	private:
		SceneDrawer mSceneDrawer;
		CubemapDrawer mCubemapDrawer;
		MeshDrawer mMeshDrawer;
		MaterialDrawer mMaterialDrawer;
		ShaderDrawer mShaderDrawer;
		Texture2DDrawer mTexture2DDrawer;
		RenderTextureDrawer mRenderTextureDrawer;

		TransformDrawer mTransformDrawer;
		RigidbodyDrawer mRigidbodyDrawer;
		CameraDrawer mCameraDrawer;
		MeshRendererDrawer mMeshRendererDrawer;
		LineRendererDrawer mLineRendererDrawer;
		LightDrawer mLightDrawer;
		BoxColliderDrawer mBoxColliderDrawer;
		SphereColliderDrawer mSphereColliderDrawer;
		CapsuleColliderDrawer mCapsuleColliderDrawer;
		MeshColliderDrawer mMeshColliderDrawer;
		TerrainDrawer mTerrainDrawer;

		bool mOpen;

	public:
		Inspector();
		~Inspector();
		Inspector(const Inspector& other) = delete;
		Inspector& operator=(const Inspector& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);

	private:
		void drawEntity(Clipboard& clipboard);
	};
} // namespace PhysicsEditor

#endif