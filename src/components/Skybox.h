#ifndef __SKYBOX_H__
#define __SKYBOX_H__

#include "Component.h"

#include "../graphics/Material.h"
#include "../graphics/Buffer.h"
#include "../graphics/VertexArrayObject.h"

namespace PhysicsEngine
{
	class Skybox : public Component
	{
		private:
			Material *material;

			Buffer skyboxVBO;
			VertexArrayObject skyboxVAO;

			static std::vector<float> triangles;

		public:
			Skybox();
			Skybox(Entity *entity);
			~Skybox();

			Material* getMaterial();

			void setMaterial(Material *material);

			void draw();
	};
}

#endif