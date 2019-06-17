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
#include "VertexBuffer.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class ForwardRenderer
	{
		private:
			World* world;
			Camera* camera;

			// main fbo
			GLuint fbo;
			GLuint color;
			GLuint position;
			GLuint normal;
			GLuint depth;
			Shader mainShader;  // whats a good name for this shader which fills depth, normals, and position? geometryShader? forwardGbufferShader?

			// ssao fbo
			GLuint ssaoFBO;
			GLuint ssaoColor;
			Shader ssaoShader;

			// directional light cascade shadow map data
			GLuint shadowCascadeFBO[5];
			GLuint shadowCascadeDepth[5];
			float cascadeEnds[6];
			glm::mat4 cascadeOrthoProj[5];
			glm::mat4 cascadeLightView[5];
			Shader depthShader;

			// spotlight shadow map data
			GLuint shadowSpotlightFBO;
			GLuint shadowSpotlightDepth;
			glm::mat4 shadowViewMatrix;
			glm::mat4 shadowProjMatrix;

			// pointlight cubemap shadow map data
			GLuint shadowCubemapFBO;
			GLuint shadowCubemapDepth;
			glm::mat4 cubeViewProjMatrices[6];
			Shader depthCubemapShader;

			// quad
			GLuint quadVAO;
			GLuint quadVBO;
			Shader quadShader;

			BatchManager batchManager;
			MeshBuffer meshBuffer;
			std::vector<RenderObject> renderObjects;

			// internal graphics state
			GraphicsCameraState cameraState; 
			GraphicsLightState lightState; 

			// timing and debug
			GraphicsQuery query;  
			GraphicsDebug debug;

			unsigned int pass;

		public:
			ForwardRenderer();
			~ForwardRenderer();

			void init(World* world);
			void update(Input input);

			GraphicsQuery getGraphicsQuery();
			GraphicsDebug getGraphicsDebug();

		private:
			void renderDebug(int view);

			void beginFrame(Camera* camera, GLuint fbo);
			void endFrame(GLuint tex);
			void cullingPass();
			void lightPass(Light* light);
			void debugPass();
			void initRenderObjectsList();
			void updateRenderObjectsList();
			void addToRenderObjectsList(MeshRenderer* meshRenderer);
			void removeFromRenderObjectsList(MeshRenderer* meshRenderer); 

			void createTextures();
			void createShaderPrograms();
			void createMeshBuffers();
			void createMainFBO();
			void createSSAOFBO();
			void createShadowMapFBOs();
			void calcShadowmapCascades(float nearDist, float farDist);
			void calcCascadeOrthoProj(glm::mat4 view, glm::vec3 direction);
			void calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection);

			void initCameraUniformState();
			void initLightUniformState();
			void updateCameraUniformState(Camera* camera);
			void updateLightUniformState(Light* light);
	};
}

#endif