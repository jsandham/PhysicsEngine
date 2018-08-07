#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>

#include <windows.h>

#include <GL/glew.h>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Shader
	{
		private:
			GLint success;
			GLuint Program;
			std::vector<std::string> attributes;
			std::vector<std::string> uniforms;
			std::vector<GLenum> attributeTypes;
			std::vector<GLenum> uniformTypes;

			std::string vertexShaderPath;
			std::string fragmentShaderPath;
			std::string geometryShaderPath;

		public:
			Shader();
			Shader(std::string vertexShaderPath, std::string fragmentShaderPath, std::string geometryShaderPath = std::string());
			~Shader();

			void bind();
			void unbind();
			bool isCompiled();
			bool compile();
			bool compile(std::string vertexShaderPath, std::string fragmentShaderPath, std::string geometryShaderPath = std::string());

			GLint getAttributeLocation(const char *name);
			GLint getUniformLocation(const char *name);
			GLuint getUniformBlockIndex(const char *name);

			std::vector<std::string> getAttributes();
			std::vector<std::string> getUniforms();
			std::vector<GLenum> getAttributeTypes();
			std::vector<GLenum> getUniformTypes();

			std::string getVertexShaderName();
			std::string getFragmentShaderName();
			std::string getGeometryShaderName();

			void setBool(std::string name, bool value);
			void setInt(std::string name, int value);
			void setFloat(std::string name, float value);
			void setVec2(std::string name, glm::vec2 &vec);
			void setVec3(std::string name, glm::vec3 &vec);
			void setVec4(std::string name, glm::vec4 &vec);
			void setMat2(std::string name, glm::mat2 &mat);
			void setMat3(std::string name, glm::mat3 &mat);
			void setMat4(std::string name, glm::mat4 &mat);

			void setUniformBlock(std::string name, GLuint bindingPoint);
	};
}

#endif