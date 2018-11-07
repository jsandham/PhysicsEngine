#ifndef __LOGICSYSTEM_H__
#define __LOGICSYSTEM_H__

#include <systems/System.h>

#include <core/Material.h>
#include <core/Shader.h>

namespace PhysicsEngine
{
	class LogicSystem : public System
	{
		public:
			Material* lineMaterial;
			Shader* lineShader;

		public:
			LogicSystem();
			LogicSystem(unsigned char* data);
			~LogicSystem();

			void init();
			void update();
	};
}

#endif