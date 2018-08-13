#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	class Shader
	{
		public:
			int shaderId;
			int globalIndex;

			std::string vertexShader;
			std::string fragmentShader;
			std::string geometryShader;

		private:
			bool programCompiled;
			GLHandle handle;

		public:
			Shader();
			~Shader();

			bool isCompiled();
			void compile();
			GLHandle getHandle();
	};
}

#endif