#ifndef __DEFERREDRENDERER_H__
#define __DEFERREDRENDERER_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Input.h"

#include "Graphics.h"
#include "GraphicsQuery.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class DeferredRenderer
	{
		private:
			World* world;

			MeshBuffer meshBuffer;
			std::vector<RenderObject> renderObjects;

			GBuffer gbuffer;

			GraphicsQuery query;

			std::vector<unsigned char> data;

		public:
			DeferredRenderer();
			~DeferredRenderer();

			void init(World* world);
			void update(Input input);
			void sort();
			void add(MeshRenderer* meshRenderer);
			void remove(MeshRenderer* meshRenderer);
	};
}

#endif