#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "../glm/glm.hpp"

#include "Guid.h"
#include "Asset.h"

#include "../graphics/GraphicsHandle.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct ShaderHeader
	{
		Guid shaderId;
		size_t vertexShaderSize;
		size_t geometryShaderSize;
		size_t fragmentShaderSize;
		size_t numberOfShaderUniforms;
	};
#pragma pack(pop)


	enum ShaderVariant
	{
		None = 0,
		Directional = 1,
		Spot = 2, 
		Point = 4,
		HardShadows = 8,
		SoftShadows = 16,
		SSAO = 32,
		Cascade = 64
	};

	enum ShaderVersion
	{
		GL330,
		GL430
	};

	struct ShaderProgram
	{
		ShaderVersion version;
		int variant;
		GLuint handle;
		bool compiled;
	};

	struct ShaderUniform
	{
		char data[64];
		char name[32]; //variable name in GLSL (including block name if applicable)
		char shortName[32]; //variable name in GLSL (excluding block name if applicable)
		char blockName[32]; //block name (empty string if not part of block)
		size_t nameLength; 
		size_t size; // size of the uniform
		GLenum type; // type of the uniform (float, vec3 or mat4, etc)
		int variant; // variant this uniform occurs in
		int location; //uniform location in shader program
		size_t index; // what index in array of uniforms we are at
		bool isEditorExposed; //if uniform can be shown in editor inspector
	};

	struct ShaderAttribute
	{
		char name[32];
	};

	class Shader : public Asset
	{
		private:
			std::string vertexShader;
			std::string fragmentShader;
			std::string geometryShader;

			bool allProgramsCompiled;
			int activeProgram;
			std::vector<ShaderProgram> programs;
			std::vector<ShaderUniform> uniforms; //uniforms need to be serialized on the material not the shader as different materials using the same shader might have different values set
			std::vector<ShaderAttribute> attributes;

		public:
			Shader();
			Shader(std::vector<char> data);
			~Shader();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool isCompiled() const;
			bool contains(int variant) const;
			void add(int variant);
			void remove(int variant);

			void compile();
			void use(int program);
			void unuse();
			void setVertexShader(const std::string vertexShader);
			void setGeometryShader(const std::string geometryShader);
			void setFragmentShader(const std::string fragmentShader);
			void setUniformBlock(const std::string& blockName, int bindingPoint) const;
			int findUniformLocation(const std::string& name, int program) const;
			int getProgramFromVariant(int variant) const;

			std::vector<ShaderUniform> getUniforms() const;
			std::vector<ShaderAttribute> getAttributeNames() const;

			void setBool(const std::string& name, bool value) const;
			void setInt(const std::string& name, int value) const;
			void setFloat(const std::string& name, float value) const;
			void setVec2(const std::string& name, const glm::vec2 &vec) const;
			void setVec3(const std::string& name, const glm::vec3 &vec) const;
			void setVec4(const std::string& name, const glm::vec4 &vec) const;
			void setMat2(const std::string& name, const glm::mat2 &mat) const;
			void setMat3(const std::string& name, const glm::mat3 &mat) const;
			void setMat4(const std::string& name, const glm::mat4 &mat) const;

			void setBool(int nameLocation, bool value) const;
			void setInt(int nameLocation, int value) const;
			void setFloat(int nameLocation, float value) const;
			void setVec2(int nameLocation, const glm::vec2 &vec) const;
			void setVec3(int nameLocation, const glm::vec3 &vec) const;
			void setVec4(int nameLocation, const glm::vec4 &vec) const;
			void setMat2(int nameLocation, const glm::mat2 &mat) const;
			void setMat3(int nameLocation, const glm::mat3 &mat) const;
			void setMat4(int nameLocation, const glm::mat4 &mat) const;

			bool getBool(const std::string& name) const;
			int getInt(const std::string& name) const;
			float getFloat(const std::string& name) const;
			glm::vec2 getVec2(const std::string& name) const;
			glm::vec3 getVec3(const std::string& name) const;
			glm::vec4 getVec4(const std::string& name) const;
			glm::mat2 getMat2(const std::string& name) const;
			glm::mat3 getMat3(const std::string& name) const;
			glm::mat4 getMat4(const std::string& name) const;

			bool getBool(int nameLocation) const;
			int getInt(int nameLocation) const;
			float getFloat(int nameLocation) const;
			glm::vec2 getVec2(int nameLocation) const;
			glm::vec3 getVec3(int nameLocation) const;
			glm::vec4 getVec4(int nameLocation) const;
			glm::mat2 getMat2(int nameLocation) const;
			glm::mat3 getMat3(int nameLocation) const;
			glm::mat4 getMat4(int nameLocation) const;



			// idea...
			//Should we have setIntOnActive etc for setting on active shader variant (using internally by engine primarily)
			//and then have setInt etc that sets on all variants used by editor and user runtime code? Maybe that too slow though for 
			//setting on all variants...maybe the eitor/user passes which variant?? Or we could leave the setInt methods as they are and 
			//add seIntOnAll methods for use by editor and user? Or setIntOnVariant?...Or maybe what we need to do is have set methods that 
			//just write to the serialized uniforms array and then when we internally call use(variant) it will apply these serialized values to
			//the now active shader?? Maybe this means we should only store user defined uniforms in uniforms array and skip things like the model 
			//matrix and lighting uniforms as these are set internally by the engine...but hen how do I distinguish what uniforms are user ones and 
			//which are internal ones??
	};

	template <>
	const int AssetType<Shader>::type = 0;

	template <typename T>
	struct IsShader { static bool value; };

	template <typename T>
	bool IsShader<T>::value = false;

	template<>
	bool IsShader<Shader>::value = true;
	template<>
	bool IsAsset<Shader>::value = true;




	/*template<int T>
	struct UniformDataType {
		static ShaderDataType dataType;
	};

	template<int T>
	ShaderDataType UniformDataType<T>::dataType = ShaderDataType::GLIntVec1;

	template<>
	ShaderDataType UniformDataType<GL_INT>::dataType = ShaderDataType::GLIntVec1;
	template<>
	ShaderDataType UniformDataType<GL_INT_VEC2>::dataType = ShaderDataType::GLIntVec2;
	template<>
	ShaderDataType UniformDataType<GL_INT_VEC3>::dataType = ShaderDataType::GLIntVec3;
	template<>
	ShaderDataType UniformDataType<GL_INT_VEC4>::dataType = ShaderDataType::GLIntVec4;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT>::dataType = ShaderDataType::GLFloatVec1;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_VEC2>::dataType = ShaderDataType::GLFloatVec2;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_VEC3>::dataType = ShaderDataType::GLFloatVec3;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_VEC4>::dataType = ShaderDataType::GLFloatVec4;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_MAT2>::dataType = ShaderDataType::GLFloatMat2;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_MAT3>::dataType = ShaderDataType::GLFloatMat3;
	template<>
	ShaderDataType UniformDataType<GL_FLOAT_MAT4>::dataType = ShaderDataType::GLFloatMat4;
	template<>
	ShaderDataType UniformDataType<GL_SAMPLER_2D>::dataType = ShaderDataType::GLSampler2D;
	template<>
	ShaderDataType UniformDataType<GL_SAMPLER_CUBE>::dataType = ShaderDataType::GLSamplerCube;*/
}

#endif