#include"../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/DeferredRendererPasses.h"

using namespace PhysicsEngine;

DeferredRenderer::DeferredRenderer()
{

}

DeferredRenderer::~DeferredRenderer()
{

}

void DeferredRenderer::init(World* world, bool renderToScreen)
{
	mWorld = world;
	mState.mRenderToScreen = renderToScreen;

	initializeDeferredRenderer(mWorld, &mState);
}

void DeferredRenderer::update(Input input, Camera* camera, std::vector<RenderObject>& renderObjects)
{
	beginDeferredFrame(mWorld, camera, &mState);

	geometryPass(mWorld, camera, &mState, renderObjects);
	lightingPass(mWorld, camera, &mState, renderObjects);
	
	endDeferredFrame(mWorld, camera, &mState);
}