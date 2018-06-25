#ifndef __FESOLID_H__
#define __FESOLID_H__

#include <vector>

#include "Component.h"

#include "../graphics/Buffer.h"
#include "../graphics/VertexArrayObject.h"

namespace PhysicsEngine
{
	class FESolid : public Component
	{
		public:
			float c;        //specific heat coefficient                         
		    float rho;      //density                            
		    float Q;        //internal heat generation   
		    float k;        //thermal conductivity coefficient

			std::vector<float> vertices;
			std::vector<int> connect;
			std::vector<int> bconnect;
			std::vector<int> groups;

			Buffer vbo;
			VertexArrayObject vao;

		public:
			FESolid();
			FESolid(Entity *entity);
			~FESolid();
	};
}

#endif