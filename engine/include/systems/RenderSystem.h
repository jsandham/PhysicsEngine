#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../graphics/GLFramebuffer.h" 
#include "../graphics/GLState.h"
#include "../graphics/GLHandle.h"

#include "../core/Material.h"
#include "../core/Shader.h"
#include "../core/Texture2D.h"
#include "../core/PerformanceGraph.h"
#include "../core/DebugWindow.h"
#include "../core/SlabBuffer.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			unsigned int pass;

			// internal graphics state
			GLCamera cameraState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

			PerformanceGraph* graph;
			DebugWindow* debugWindow;

			Material* graphMaterial;
			Material* windowMaterial;
			Material* normalMapMaterial;
			Material* depthMapMaterial;
			Material* lineMaterial;

			Shader* graphShader;
			Shader* windowShader;
			Shader* normalMapShader;
			Shader* depthMapShader;
			Shader* lineShader;

			Material* debugMaterial;
			Texture2D* debugBuffer;

			GLFramebuffer fbo;


			SlabBuffer lineBuffer;

		public:
			RenderSystem();
			RenderSystem(unsigned char* data);
			~RenderSystem();

			void init();
			void update();

		private:
			void renderScene();
			void renderScene(Material* material);
	};
}

#endif