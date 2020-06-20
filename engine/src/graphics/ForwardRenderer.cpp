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

void ForwardRenderer::update(Input input, Camera* camera, std::vector<RenderObject>& renderObjects)
{
	const PoolAllocator<Light>* lightAllocator = mWorld->getComponentAllocator_Const<Light>();

	beginFrame(mWorld, camera, &mState);

	if (camera->mSSAO == CameraSSAO::SSAO_On) {
		computeSSAO(mWorld, camera, &mState, renderObjects);
	}

	for (int j = 0; j < mWorld->getNumberOfComponents<Light>(lightAllocator); j++) {
		Light* light = mWorld->getComponentByIndex<Light>(lightAllocator, j);
		Transform* lightTransform = light->getComponent<Transform>(mWorld);

		renderShadows(mWorld, camera, light, lightTransform, &mState, renderObjects);
		renderOpaques(mWorld, camera, light, lightTransform, &mState, renderObjects);
		renderTransparents();
	}

	renderColorPicking(mWorld, camera, &mState, renderObjects);

	postProcessing();
	endFrame(mWorld, camera, &mState);
}