#ifndef __SLAB_BUFFER_H__
#define __SLAB_BUFFER_H__

#include "Line.h"
#include "Material.h"

#include "../graphics/GLHandle.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

namespace PhysicsEngine
{
	struct SlabNode  //GLLineBufferNode??
	{
		SlabNode* next;
		int numberOfLinesToDraw;
		Material* material;
		GLHandle nodeVAO;
		GLHandle vertexVBO;
		float buffer[300000]; //vertices??
	};

	class SlabBuffer //LineBuffer??
	{
		private:
			SlabNode* root;
			SlabNode* next;

			static int test;

		public:
			SlabBuffer();
			~SlabBuffer();

			void clear();
			void add(glm::vec3 start, glm::vec3 end, Material* material);
			void add(std::vector<float> lines, Material* material);

			bool hasNext();
			SlabNode* getNext();

	};
}

#endif