#include "../../include/core/Shader.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

std::string Shader::lineVertexShader = "#version 330 core\n"
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * vec4(position, 1.0);\n"
"}";

std::string Shader::lineFragmentShader = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

std::string Shader::graphVertexShader = "#version 330 core\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"}";

std::string Shader::graphFragmentShader = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

std::string Shader::normalMapVertexShader = "#version 330 core\n"
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"out vec3 Normal;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * vec4(position, 1.0);\n"
"   Normal = normal;\n"
"}";

std::string Shader::normalMapFragmentShader = "#version 330 core\n"
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(Normal.xyz, 1.0f);\n"
"}";





Shader::Shader()
{
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
