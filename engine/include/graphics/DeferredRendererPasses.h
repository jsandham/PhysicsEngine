#ifndef __DEFERRED_RENDERER_PASSES_H__
#define __DEFERRED_RENDERER_PASSES_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"

#include "DeferredRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	void initializeDeferredRenderer(World* world, DeferredRendererState* state);

	void beginDeferredFrame(World* world, Camera* camera, DeferredRendererState* state);

	void geometryPass(World* world, 
					  Camera* camera, 
					  DeferredRendererState* state, 
					  const std::vector<RenderObject>& renderObjects);

	void lightingPass(World* world, 
					  Camera* camera, 
					  DeferredRendererState* state, 
					  const std::vector<RenderObject>& renderObjects);

	void endDeferredFrame(World* world, Camera* camera, DeferredRendererState* state);
}

#endif