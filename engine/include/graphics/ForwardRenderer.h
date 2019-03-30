#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <map>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Guid.h"

#include "BatchManager.h"
#include "GraphicsState.h"
#include "GraphicsQuery.h"
#include "GraphicsDebug.h"

namespace PhysicsEngine
{
	// where should this live? Graphics? GLState? 
	struct InternalMesh  //OpenGLMesh? InternalMesh? DynamicMesh? Maybe I should put this back in Mesh and just not use it when batching?
	{
		GLuint VAO;
		GLuint vertexVBO;
		GLuint normalVBO;
		GLuint texCoordVBO;
	};


	class ForwardRenderer
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

			// maybe move these into render system istead and pass them in by pointer to be updated
			GraphicsQuery query;  
			GraphicsDebug debug;

			unsigned int pass;

		public:
			ForwardRenderer();
			~ForwardRenderer();

			void init(World* world);
			void update();

			GraphicsQuery getGraphicsQuery();
			GraphicsDebug getGraphicsDebug();

		private:
			void render();
			void renderDebug();
	};
}

#endif