#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../components/Transform.h"
#include "../components/MeshRenderer.h"
#include "../components/DirectionalLight.h"
#include "../components/SpotLight.h"
#include "../components/PointLight.h"
#include "../components/Camera.h"

#include "../core/Texture2D.h"
#include "../core/Cubemap.h"

#include "../graphics/GLState.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			unsigned int numLights;
			unsigned int pass;

			// camera data   //  Probably can get rid of this now that it is contained inside GLCamera???
			glm::mat4 view;
			glm::mat4 projection;
			glm::vec3 cameraPos;

			// shadow textures and framebuffer
			std::vector<Texture2D*> cascadeTexture2D;
			Texture2D* shadowTexture2D;
			Cubemap* shadowCubemap;
			//Framebuffer* shadowFBO;

			// internal graphics state
			GLCamera cameraState;
			GLShadow shadowState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

			std::vector<glm::mat4> cascadeLightView;
			std::vector<glm::mat4> cascadeOrthoProj;
			std::vector<glm::mat4> cubeViewMatrices;

		public:
			// RenderSystem(Manager *manager, SceneContext* context);
			RenderSystem();
			~RenderSystem();

			size_t getSize();
			void init();
			void update();

		private:
			void createShadowMaps();
			void createEnvironmentMap();
			void calcCascadeOrthoProj(glm::vec3 lightDirection);
			void calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection);

			void renderScene();
			void renderShadowMap(Texture2D* texture, glm::mat4 lightView, glm::mat4 lightProjection);
			//void renderDepthCubemap(Cubemap* cubemap, glm::mat4 lightProjection);
	};
}

#endif


// #ifndef __RENDERSYSTEM_H__
// #define __RENDERSYSTEM_H__

// #include <vector>

// #include <GL/glew.h>
// #include <gl/gl.h>

// #include "System.h"

// #include "../components/Transform.h"
// #include "../components/MeshRenderer.h"
// #include "../components/DirectionalLight.h"
// #include "../components/SpotLight.h"
// #include "../components/PointLight.h"
// #include "../components/Camera.h"

// #include "../graphics/Framebuffer.h"
// #include "../graphics/Buffer.h"
// #include "../graphics/GraphicState.h"

// #define GLM_FORCE_RADIANS

// #include "../glm/glm.hpp"
// #include "../glm/gtc/type_ptr.hpp"
// #include "../glm/gtc/matrix_transform.hpp"

// namespace PhysicsEngine
// {
// 	class RenderSystem : public System
// 	{
// 		private:
// 			unsigned int numLights;
// 			unsigned int pass;

// 			// camera data
// 			glm::mat4 view;
// 			glm::mat4 projection;
// 			glm::vec3 cameraPos;

// 			// shadow textures and framebuffer
// 			std::vector<Texture2D*> cascadeTexture2D;
// 			Texture2D* shadowTexture2D;
// 			Cubemap* shadowCubemap;
// 			Framebuffer* shadowFBO;

// 			// internal shaders
// 			Shader depthShader;
// 			Shader particleShader;

// 			// internal graphics state
// 			GraphicState state;

// 			std::vector<VertexArrayObject> meshVAO;
// 			std::vector<Buffer> vertexVBO;
// 			std::vector<Buffer> normalVBO;
// 			std::vector<Buffer> texCoordVBO;

// 			std::vector<glm::mat4> cascadeLightView;
// 			std::vector<glm::mat4> cascadeOrthoProj;
// 			std::vector<glm::mat4> cubeViewMatrices;

// 		public:
// 			RenderSystem(Manager *manager);
// 			~RenderSystem();

// 			void init();
// 			void update();

// 		private:
// 			void createShadowMaps();
// 			void createEnvironmentMap();
// 			void calcCascadeOrthoProj(glm::vec3 lightDirection);
// 			void calcCubeViewMatrices(glm::vec3 lightPosition, glm::mat4 lightProjection);

// 			void renderScene();
// 			void renderShadowMap(Texture2D* texture, glm::mat4 lightView, glm::mat4 lightProjection);
// 			void renderDepthCubemap(Cubemap* cubemap, glm::mat4 lightProjection);
// 	};
// }

// #endif