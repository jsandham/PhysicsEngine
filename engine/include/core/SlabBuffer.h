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
	struct SlabNode  
	{
		SlabNode* next;
		size_t count;
		size_t size;
		Material* material;
		GLHandle vao;
		GLHandle vbo;
		float* buffer;
	};

	class SlabBuffer
	{
		private:
			SlabNode* root;
			SlabNode* next;
			size_t blockSize;

			static int test;

		public:
			SlabBuffer(size_t blockSize);
			~SlabBuffer();

			void clear();
			void add(std::vector<float> data, Material* material);

			bool hasNext();
			SlabNode* getNext();
			size_t getBlockSize();
	};
}

#endif