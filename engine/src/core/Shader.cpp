#include "../../include/core/Shader.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Shader::Shader()
{
	shaderId = -1;
	//globalIndex = -1;
	programCompiled = false;
}

Shader::~Shader()
{

}

bool Shader::isCompiled()
{
	return programCompiled;
}

void Shader::compile()
{
	Graphics::compile(this);
}

void Shader::setBool(std::string name, bool value)
{
	Graphics::setBool(this, name, value);
}

void Shader::setInt(std::string name, int value)
{
	Graphics::setInt(this, name, value);
}

void Shader::setFloat(std::string name, float value)
{
	Graphics::setFloat(this, name, value);
}

void Shader::setVec2(std::string name, glm::vec2 &vec)
{
	Graphics::setVec2(this, name, vec);
}

void Shader::setVec3(std::string name, glm::vec3 &vec) 
{
	Graphics::setVec3(this, name, vec);
}

void Shader::setVec4(std::string name, glm::vec4 &vec)
{
	Graphics::setVec4(this, name, vec);
}

void Shader::setMat2(std::string name, glm::mat2 &mat)
{
	Graphics::setMat2(this, name, mat);
}

void Shader::setMat3(std::string name, glm::mat3 &mat)
{
	Graphics::setMat3(this, name, mat);
}

void Shader::setMat4(std::string name, glm::mat4 &mat)
{
	Graphics::setMat4(this, name, mat);
}
