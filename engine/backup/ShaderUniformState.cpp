#include "../../include/graphics/ShaderUniformState.h"

using namespace PhysicsEngine;

ShaderUniformState::ShaderUniformState(Shader *shader) : shader(shader)
{

}

ShaderUniformState::~ShaderUniformState()
{

}

void ShaderUniformState::setUniforms()
{
	std::map<std::string, bool>::iterator it1;
	for (it1 = boolUniforms.begin(); it1 != boolUniforms.end(); it1++){
		shader->setBool(it1->first, it1->second);
	}

	std::map<std::string, int>::iterator it2;
	for (it2 = intUniforms.begin(); it2 != intUniforms.end(); it2++){
		shader->setInt(it2->first, it2->second);
	}

	std::map<std::string, float>::iterator it3;
	for (it3 = floatUniforms.begin(); it3 != floatUniforms.end(); it3++){
		shader->setFloat(it3->first, it3->second);
	}

	std::map<std::string, glm::vec2>::iterator it4;
	for (it4 = vec2Uniforms.begin(); it4 != vec2Uniforms.end(); it4++){
		shader->setVec2(it4->first, it4->second);
	}

	std::map<std::string, glm::vec3>::iterator it5;
	for (it5 = vec3Uniforms.begin(); it5 != vec3Uniforms.end(); it5++){
		shader->setVec3(it5->first, it5->second);
	}

	std::map<std::string, glm::vec4>::iterator it6;
	for (it6 = vec4Uniforms.begin(); it6 != vec4Uniforms.end(); it6++){
		shader->setVec4(it6->first, it6->second);
	}

	std::map<std::string, glm::mat2>::iterator it7;
	for (it7 = mat2Uniforms.begin(); it7 != mat2Uniforms.end(); it7++){
		shader->setMat2(it7->first, it7->second);
	}

	std::map<std::string, glm::mat3>::iterator it8;
	for (it8 = mat3Uniforms.begin(); it8 != mat3Uniforms.end(); it8++){
		shader->setMat3(it8->first, it8->second);
	}

	std::map<std::string, glm::mat4>::iterator it9;
	for (it9 = mat4Uniforms.begin(); it9 != mat4Uniforms.end(); it9++){
		shader->setMat4(it9->first, it9->second);
	}
}

void ShaderUniformState::clear()
{
	boolUniforms.clear();
	intUniforms.clear();
	floatUniforms.clear();
	vec2Uniforms.clear();
	vec3Uniforms.clear();
	vec4Uniforms.clear();
	mat2Uniforms.clear();
	mat3Uniforms.clear();
	mat4Uniforms.clear();
}

void ShaderUniformState::setBool(std::string name, bool value)
{
	boolUniforms[name] = value;
}

void ShaderUniformState::setInt(std::string name, int value)
{
	intUniforms[name] = value;
}

void ShaderUniformState::setFloat(std::string name, float value)
{
	floatUniforms[name] = value;
}

void ShaderUniformState::setVec2(std::string name, glm::vec2 &vec)
{
	vec2Uniforms[name] = vec;
}

void ShaderUniformState::setVec3(std::string name, glm::vec3 &vec)
{
	vec3Uniforms[name] = vec;
}

void ShaderUniformState::setVec4(std::string name, glm::vec4 &vec)
{
	vec4Uniforms[name] = vec;
}

void ShaderUniformState::setMat2(std::string name, glm::mat2 &mat)
{
	mat2Uniforms[name] = mat;
}

void ShaderUniformState::setMat3(std::string name, glm::mat3 &mat)
{
	mat3Uniforms[name] = mat;
}

void ShaderUniformState::setMat4(std::string name, glm::mat4 &mat)
{
	mat4Uniforms[name] = mat;
}
