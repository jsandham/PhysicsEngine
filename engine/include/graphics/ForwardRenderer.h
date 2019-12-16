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
#include "FramebufferData.h"

namespace PhysicsEngine
{
	class ForwardRenderer
	{
	private:
		World* world;

		std::vector<RenderObject> renderObjects;

		FramebufferData fboData;
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















































//#ifndef __FORWARDRENDERER_H__
//#define __FORWARDRENDERER_H__
//
//#include <map>
//#include <vector>
//#include <GL/glew.h>
//#include <gl/gl.h>
//
//#include "../core/World.h"
//#include "../core/Guid.h"
//#include "../core/Input.h"
//
//#include "../components/Light.h"
//#include "../components/MeshRenderer.h"
//
//#include "BatchManager.h"
//#include "GraphicsState.h"
//#include "GraphicsQuery.h"
//#include "GraphicsDebug.h"
//#include "GraphicsTargets.h"
//#include "RenderObject.h"
//#include "ShadowMapData.h"
//#include "FramebufferData.h"
//
//namespace PhysicsEngine
//{
//	class ForwardRenderer
//	{
//		private:
//			World* world;
//			Camera* camera;
//
//			FramebufferData fboData;
//
//			// shadow map data
//			ShadowMapData shadowMapData;
//
//			std::vector<RenderObject> renderObjects;
//
//			// internal graphics state
//			GraphicsCameraState cameraState; 
//			GraphicsLightState lightState; 
//
//			// timing and debug
//			GraphicsQuery query;  
//			GraphicsDebug debug;
//			GraphicsTargets targets;
//
//			unsigned int pass;
//			bool renderToScreen;
//
//		public:
//			ForwardRenderer();
//			~ForwardRenderer();
//
//			void init(World* world, bool renderToScreen);
//			void update(Input input);
//
//			GraphicsQuery getGraphicsQuery() const;
//			GraphicsDebug getGraphicsDebug() const;
//			GraphicsTargets getGraphicsTargets() const;
//
//		private:
//			void beginFrame(Camera* camera, GLuint fbo);
//			void endFrame();
//			void cullingPass();
//			void lightPass(Light* light);
//			void debugPass();
//			void updateRenderObjectsList();
//			void addToRenderObjectsList(MeshRenderer* meshRenderer);
//			void removeFromRenderObjectsList(MeshRenderer* meshRenderer); 
//
//			void updateAssetsInRenderer();
//			void createTextures();
//			void createShaderPrograms();
//			void createInternalShaderPrograms();
//			void createMeshBuffers();
//			void createMainFBO();
//			void createSSAOFBO();
//			void createShadowMapFBOs();
//			void calcShadowmapCascades(float nearDist, float farDist);
//			void calcCascadeOrthoProj(glm::mat4 view, glm::vec3 direction);
//			void calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection);
//
//			void initCameraUniformState();
//			void initLightUniformState();
//			void updateCameraUniformState(Camera* camera);
//			void updateLightUniformState(Light* light);
//	};
//}
//
//#endif