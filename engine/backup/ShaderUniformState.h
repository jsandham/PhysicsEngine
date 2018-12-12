#ifndef __SHADERUNIFORMSTATE_H__
#define __SHADERUNIFORMSTATE_H__

#include <string>
#include <map>

#include "Shader.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class ShaderUniformState
	{
		private:
			Shader *shader;

			std::map<std::string, bool> boolUniforms;
			std::map<std::string, int> intUniforms;
			std::map<std::string, float> floatUniforms;
			std::map<std::string, glm::vec2> vec2Uniforms;
			std::map<std::string, glm::vec3> vec3Uniforms;
			std::map<std::string, glm::vec4> vec4Uniforms;
			std::map<std::string, glm::mat2> mat2Uniforms;
			std::map<std::string, glm::mat3> mat3Uniforms;
			std::map<std::string, glm::mat4> mat4Uniforms;

		public:
			ShaderUniformState(Shader *shader);
			~ShaderUniformState();

			void setUniforms();
			void clear();

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