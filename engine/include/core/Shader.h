#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>

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

		public:
			Shader();
			~Shader();
	};
}

#endif