#ifndef __CLOTH_H__
#define __CLOTH_H__

#include <vector>

#include "Component.h"

#include "../graphics/Buffer.h"
#include "../graphics/VertexArrayObject.h"

namespace PhysicsEngine
{
	class Cloth : public Component
	{
		public:
			int nx;
			int ny;
			std::vector<float> particles;
			std::vector<int> particleTypes;

			float kappa;            //spring stiffness coefficient
			float c;                //spring dampening coefficient
			float mass;             //mass

			Buffer vertexVBO;
			Buffer normalVBO;
			VertexArrayObject clothVAO;

		public:
			Cloth();
			~Cloth();
	};
}

#endif