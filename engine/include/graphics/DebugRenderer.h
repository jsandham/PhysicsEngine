#ifndef __DEBUG_RENDERER_H__
#define __DEBUG_RENDERER_H__

#include "../core/World.h"
#include "../core/Shader.h"
#include "../core/Material.h"
#include "../core/Texture2D.h"
#include "../core/Font.h"
#include "../core/PerformanceGraph.h"
#include "../core/DebugWindow.h"
#include "../core/SlabBuffer.h"

#include "../graphics/GLFramebuffer.h"

namespace PhysicsEngine
{
	class DebugRenderer
	{
		private:
			World* world;

			PerformanceGraph* graph;
			DebugWindow* debugWindow;
			SlabBuffer* lineBuffer;

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
			Shader* fontShader;

			Font* font;

			Material* debugMaterial;
			Texture2D* debugBuffer;

			GLFramebuffer fbo;

			GLuint vao;
			GLuint vbo;

		public:
			DebugRenderer();
			~DebugRenderer();

			void init(World* world);
			void update();
	};
}

#endif