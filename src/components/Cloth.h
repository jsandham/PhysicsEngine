#ifndef __CLOTH_H__
#define __CLOTH_H__

#include <vector>

#include "Component.h"

// #include "../graphics/Buffer.h"
// #include "../graphics/VertexArrayObject.h"

namespace PhysicsEngine
{
	class Cloth : public Component
	{
		public:
			int nx;
			int ny;
			std::vector<float> particles;
			std::vector<int> particleTypes;

			float kappa;
			float c;
			float mass;

			// Buffer vbo;
			// VertexArrayObject vao;

		public:
			Cloth();
			Cloth(Entity *entity);
			~Cloth();
	};
}

#endif