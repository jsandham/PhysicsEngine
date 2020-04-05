#ifndef __DEFERREDRENDERER_H__
#define __DEFERREDRENDERER_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Input.h"
#include "../components/MeshRenderer.h"

#include "Graphics.h"
#include "GraphicsQuery.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class DeferredRenderer
	{
		private:
			World* mWorld;

			/*MeshBuffer mMeshBuffer;
			std::vector<RenderObject> mRenderObjects;

			GBuffer mGbuffer;

			GraphicsQuery mQuery;

			std::vector<unsigned char> mData;*/

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