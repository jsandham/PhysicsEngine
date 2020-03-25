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
		World* mWorld;

		std::vector<RenderObject> mRenderObjects;

		ScreenData mScreenData;
		ShadowMapData mShadowMapData;

		// internal graphics state
		GraphicsCameraState mCameraState;
		GraphicsLightState mLightState;

		// timing and debug
		GraphicsQuery mQuery;
		GraphicsDebug mDebug;
		GraphicsTargets mTargets;

		bool mRenderToScreen;

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