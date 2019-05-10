#ifndef __FORWARDRENDERER_H__
#define __FORWARDRENDERER_H__

#include <map>
#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Guid.h"

#include "../components/MeshRenderer.h"

#include "BatchManager.h"
#include "GraphicsState.h"
#include "GraphicsQuery.h"
#include "GraphicsDebug.h"
#include "VertexBuffer.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	// where should this live? Graphics? GLState? 
	// struct InternalMesh  //OpenGLMesh? InternalMesh? DynamicMesh? Maybe I should put this back in Mesh and just not use it when batching?
	// {
	// 	GLuint VAO;
	// 	GLuint vertexVBO;
	// 	GLuint normalVBO;
	// 	GLuint texCoordVBO;
	// };

	class ForwardRenderer
	{
		private:
			World* world;
			Camera* camera;

			// fbo
			GLenum framebufferStatus;
			GLuint fbo;
			GLuint color;
			GLuint depth;
			Shader depthShader;

			// quad
			GLuint quadVAO;
			GLuint quadVBO;
			Shader quadShader;

			BatchManager batchManager;
			MeshBuffer meshBuffer;
			std::vector<RenderObject> renderObjects;
			//std::map<Guid, InternalMesh> meshIdToInternalMesh; 

			// internal graphics state
			GraphicsCameraState cameraState; 
			GraphicsDirectionalLightState directionLightState; 
			GraphicsSpotLightState spotLightState;
			GraphicsPointLightState pointLightState;
			GraphicsQuery query;  
			GraphicsDebug debug;

			unsigned int pass;

		public:
			ForwardRenderer();
			~ForwardRenderer();

			void init(World* world);
			void update();
			void sort();
			void add(MeshRenderer* meshRenderer);
			void remove(MeshRenderer* meshRenderer);

			GraphicsQuery getGraphicsQuery();
			GraphicsDebug getGraphicsDebug();

		private:
			void render();
			void renderDebug(int view);



			void initCameraUniformState();
			void initDirectionalLightUniformState();
			void initSpotLightUniformState();
			void initPointLightUniformState();

			void updateCameraUniformState();
			void updateDirectionalLightUniformState(DirectionalLight* light);
			void updateSpotLightUniformState(SpotLight* light);
			void updatePointLightUniformState(PointLight* light);

			void createShadowMapTextures();
	};
}

#endif