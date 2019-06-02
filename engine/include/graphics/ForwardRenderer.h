#ifndef __FORWARDRENDERER_H__
#define __FORWARDRENDERER_H__

#include <map>
#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Guid.h"
#include "../core/Input.h"

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
			GLuint depth;
			Shader depthShader;

			// directional light cascade shadow map data
			GLuint shadowCascadeFBO[5];
			GLuint shadowCascadeDepth[5];
			float cascadeEnds[6];
			glm::mat4 cascadeOrthoProj[5];
			glm::mat4 cascadeLightView[5];

			// spotlight shadow map data
			GLuint shadowSpotlightFBO;
			GLuint shadowSpotlightDepth;

			// pointlight cubemap shadow map data
			GLuint shadowCubemapFBO;
			GLuint shadowCubemapDepth;
			glm::mat4 cubeViewMatrices[6];

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
			void add(MeshRenderer* meshRenderer);
			void remove(MeshRenderer* meshRenderer);

			GraphicsQuery getGraphicsQuery();
			GraphicsDebug getGraphicsDebug();

		private:
			void render(GLuint fbo, ShaderVariant variant); //renderScene?
			void renderShadowMap(GLuint fbo, glm::mat4 lightView, glm::mat4 lightProjection);
			void renderDebug(int view);

			void beginFrame(Camera* camera, GLuint fbo);
			void endFrame(GLuint tex);
			void renderDirectionalLights();
			void renderSpotLights();
			void renderPointLights();

			void createTextures();
			void createShaderPrograms();
			void createMeshBuffers();
			void createMainFBO();
			void createShadowMapFBOs();
			void calcShadowmapCascades(float nearDist, float farDist);
			void calcCascadeOrthoProj(glm::mat4 view, glm::vec3 direction);
			void calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection);

			//void renderTextureToScreen(GLuint tex); //could call endFrame()?




			void initCameraUniformState();
			void initLightUniformState();


			void updateCameraUniformState(Camera* camera);
			void updateLightUniformState(DirectionalLight* light);
			void updateLightUniformState(SpotLight* light);
			void updateLightUniformState(PointLight* light);
	};
}

#endif