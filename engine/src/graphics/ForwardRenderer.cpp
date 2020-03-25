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
	mRenderToScreen = renderToScreen;

	initializeRenderer(world, mScreenData, mShadowMapData, mCameraState, mLightState, mDebug, mQuery);
}

void ForwardRenderer::update(Input input)
{
	registerRenderAssets(mWorld);
	registerRenderObjects(mWorld, mRenderObjects);
	registerCameras(mWorld);

	updateTransforms(mWorld, mRenderObjects);

	for (int i = 0; i < mWorld->getNumberOfComponents<Camera>(); i++) {
		Camera* camera = mWorld->getComponentByIndex<Camera>(i);

		cullRenderObjects(camera, mRenderObjects);

		beginFrame(camera, mCameraState, mLightState, mQuery);

		computeSSAO(mWorld, camera, mRenderObjects, mScreenData, mQuery);

		for (int j = 0; j < mWorld->getNumberOfComponents<Light>(); j++) {
			Light* light = mWorld->getComponentByIndex<Light>(j);

			renderShadows(mWorld, camera, light, mRenderObjects, mShadowMapData, mQuery);
			renderOpaques(mWorld, camera, light, mRenderObjects, mShadowMapData, mLightState, mQuery);
			renderTransparents();
		}

		postProcessing();
		endFrame(mWorld, camera, mRenderObjects, mScreenData, mTargets, mDebug, mQuery, mRenderToScreen);
	}
}

GraphicsQuery ForwardRenderer::getGraphicsQuery() const
{
	return mQuery;
}

GraphicsDebug ForwardRenderer::getGraphicsDebug() const
{
	return mDebug;
}

GraphicsTargets ForwardRenderer::getGraphicsTargets() const
{
	return mTargets;
}