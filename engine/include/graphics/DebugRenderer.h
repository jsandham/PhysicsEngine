#ifndef __DEBUG_RENDERER_H__
#define __DEBUG_RENDERER_H__

#include "../core/World.h"
#include "../core/Shader.h"
#include "../core/Material.h"
#include "../core/Texture2D.h"
#include "../core/Font.h"
#include "../core/Input.h"
#include "../core/SlabBuffer.h"

#include "../graphics/Graphics.h"
#include "../graphics/GraphicsQuery.h"
#include "../graphics/GraphicsDebug.h"

namespace PhysicsEngine
{
	class DebugRenderer
	{
		private:
			World* world;

			Texture2D* windowTexture;

			PerformanceGraph graph;
			DebugWindow window;
			Font font;


			SlabBuffer* lineBuffer;
			Material* lineMaterial;
			Shader* lineShader;

		public:
			DebugRenderer();
			~DebugRenderer();

			void init(World* world);
			void update(Input input, GraphicsDebug debug, GraphicsQuery query);
	};
}

#endif