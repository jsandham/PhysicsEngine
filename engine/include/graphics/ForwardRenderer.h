#ifndef __FORWARDRENDERER_H__
#define __FORWARDRENDERER_H__

#include <map>
#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Guid.h"
#include "../core/Input.h"

#include "../components/Light.h"
#include "../components/MeshRenderer.h"

#include "ForwardRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class ForwardRenderer
	{
	private:
		World* mWorld;

		// render objects 
		std::vector<RenderObject> mRenderObjects;

		// forward renderer state
		ForwardRendererState mState;

	public:
		ForwardRenderer();
		~ForwardRenderer();

		void init(World* world, bool renderToScreen);
		void update(Input input);

		GraphicsQuery getGraphicsQuery() const;
		GraphicsTargets getGraphicsTargets() const;
	};
}

#endif