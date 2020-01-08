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

#include "BatchManager.h"
#include "GraphicsState.h"
#include "GraphicsQuery.h"
#include "GraphicsDebug.h"
#include "GraphicsTargets.h"
#include "RenderObject.h"
#include "ShadowMapData.h"
#include "ScreenData.h"

namespace PhysicsEngine
{
	class ForwardRenderer
	{
	private:
		World* world;

		std::vector<RenderObject> renderObjects;

		ScreenData screenData;
		ShadowMapData shadowMapData;

		// internal graphics state
		GraphicsCameraState cameraState;
		GraphicsLightState lightState;

		// timing and debug
		GraphicsQuery query;
		GraphicsDebug debug;
		GraphicsTargets targets;

		bool renderToScreen;

	public:
		ForwardRenderer();
		~ForwardRenderer();

		void init(World* world, bool renderToScreen);
		void update(Input input);

		GraphicsQuery getGraphicsQuery() const;
		GraphicsDebug getGraphicsDebug() const;
		GraphicsTargets getGraphicsTargets() const;
	};
}

#endif