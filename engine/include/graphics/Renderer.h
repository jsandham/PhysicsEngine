#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <map>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Guid.h"

#include "GLState.h"
#include "BatchManager.h"
#include "GraphicsQuery.h"

namespace PhysicsEngine
{
	// where should this live? Graphics? GLState? 
	struct InternalMesh  //OpenGLMesh? InternalMesh? DynamicMesh?
	{
		GLuint VAO;
		GLuint vertexVBO;
		GLuint normalVBO;
		GLuint texCoordVBO;
	};


	class Renderer
	{
		private:
			World* world;

			// internal graphics state
			GLCamera cameraState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

			BatchManager batchManager;
			std::map<Guid, InternalMesh> meshIdToInternalMesh; 

			GraphicsQuery query;

			unsigned int pass;

		public:
			Renderer();
			~Renderer();

			void init(World* world);
			void update();

			GraphicsQuery getGraphicsQuery();

		private:
			void render();
	};
}

#endif