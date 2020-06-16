#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Util.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{
	
}

ForwardRenderer::~ForwardRenderer()
{

}

void ForwardRenderer::init(World* world, bool renderToScreen)
{
	mWorld = world;
	mState.mRenderToScreen = renderToScreen;
	
	initializeRenderer(mWorld, &mState);
}

void ForwardRenderer::update(Input input)
{
	registerRenderAssets(mWorld);
	registerCameras(mWorld);

	updateRenderObjects(mWorld, mRenderObjects);
	updateModelMatrices(mWorld, mRenderObjects);

	const PoolAllocator<Camera>* cameraAllocator = mWorld->getComponentAllocator_Const<Camera>();
	const PoolAllocator<Light>* lightAllocator = mWorld->getComponentAllocator_Const<Light>();

	for (int i = 0; i < mWorld->getNumberOfComponents<Camera>(cameraAllocator); i++) {
		Camera* camera = mWorld->getComponentByIndex<Camera>(cameraAllocator, i);

		cullRenderObjects(camera, mRenderObjects);

		beginFrame(mWorld, camera, &mState);

		if (camera->mSSAO == CameraSSAO::SSAO_On) {
			computeSSAO(mWorld, camera, &mState, mRenderObjects);
		}

		for (int j = 0; j < mWorld->getNumberOfComponents<Light>(lightAllocator); j++) {
			Light* light = mWorld->getComponentByIndex<Light>(lightAllocator, j);
			Transform* lightTransform = light->getComponent<Transform>(mWorld);

			renderShadows(mWorld, camera, light, lightTransform, &mState, mRenderObjects);
			renderOpaques(mWorld, camera, light, lightTransform, &mState, mRenderObjects);
			renderTransparents();
		}

		renderColorPicking(mWorld, camera, &mState, mRenderObjects);

		postProcessing();
		endFrame(mWorld, camera, &mState, mRenderObjects);
	}
}

GraphicsQuery ForwardRenderer::getGraphicsQuery() const
{
	return mState.mQuery;
}

GraphicsTargets ForwardRenderer::getGraphicsTargets() const
{
	return mState.mTargets;
}