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
#pragma pack(push, 1)
	struct ShaderHeader
	{
		Guid shaderId;
		size_t vertexShaderSize;
		size_t geometryShaderSize;
		size_t fragmentShaderSize;
	};
#pragma pack(pop)
	
	class Shader : public Asset
	{
		public:
			static std::string lineVertexShader;
			static std::string lineFragmentShader;
			static std::string graphVertexShader;
			static std::string graphFragmentShader;
			static std::string windowVertexShader;
			static std::string windowFragmentShader;
			static std::string normalMapVertexShader;
			static std::string normalMapFragmentShader;
			static std::string depthMapVertexShader;
			static std::string depthMapFragmentShader;

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