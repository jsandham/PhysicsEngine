#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>

#include "../glm/glm.hpp"

#include "Guid.h"
#include "Asset.h"

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	class Shader : public Asset
	{
		public:
			std::string vertexShader;
			std::string fragmentShader;
			std::string geometryShader;

			bool programCompiled;
			GLHandle program;

		public:
			Shader();
			~Shader();

			bool isCompiled();
			void compile();

			void setBool(std::string name, bool value);
			void setInt(std::string name, int value);
			void setFloat(std::string name, float value);
			void setVec2(std::string name, glm::vec2 &vec);
			void setVec3(std::string name, glm::vec3 &vec);
			void setVec4(std::string name, glm::vec4 &vec);
			void setMat2(std::string name, glm::mat2 &mat);
			void setMat3(std::string name, glm::mat3 &mat);
			void setMat4(std::string name, glm::mat4 &mat);
	};
}

#endif