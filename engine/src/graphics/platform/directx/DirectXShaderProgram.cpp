#include "../../../../include/graphics/platform/directx/DirectXShaderProgram.h"

using namespace PhysicsEngine;

DirectXShaderProgram::DirectXShaderProgram()
{

}

DirectXShaderProgram ::~DirectXShaderProgram()
{

}

void DirectXShaderProgram::load(const std::string &vertex, const std::string &fragment, const std::string &geometry)
{

}

void DirectXShaderProgram::load(const std::string &vertex, const std::string &fragment)
{
    this->load(vertex, fragment, "");
}

void DirectXShaderProgram::compile()
{

}

void DirectXShaderProgram::bind()
{

}

void DirectXShaderProgram::unbind()
{

}

int DirectXShaderProgram::findUniformLocation(const std::string &name) const
{
    return -1;
}

std::vector<ShaderUniform> DirectXShaderProgram::getUniforms() const
{
    return std::vector<ShaderUniform>();
}

std::vector<ShaderAttribute> DirectXShaderProgram::getAttributes() const
{
    return std::vector<ShaderAttribute>();
}

void DirectXShaderProgram::setBool(const char *name, bool value)
{

}

void DirectXShaderProgram::setInt(const char *name, int value)
{

}

void DirectXShaderProgram::setFloat(const char *name, float value)
{

}

void DirectXShaderProgram::setColor(const char *name, const Color &color)
{

}

void DirectXShaderProgram::setColor32(const char *name, const Color32 &color)
{

}

void DirectXShaderProgram::setVec2(const char *name, const glm::vec2 &vec)
{

}

void DirectXShaderProgram::setVec3(const char *name, const glm::vec3 &vec)
{

}

void DirectXShaderProgram::setVec4(const char *name, const glm::vec4 &vec)
{

}

void DirectXShaderProgram::setMat2(const char *name, const glm::mat2 &mat)
{

}

void DirectXShaderProgram::setMat3(const char *name, const glm::mat3 &mat)
{

}

void DirectXShaderProgram::setMat4(const char *name, const glm::mat4 &mat)
{

}

void DirectXShaderProgram::setTexture2D(const char *name, int texUnit, TextureHandle* tex)
{

}

void DirectXShaderProgram::setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{

}

void DirectXShaderProgram::setBool(int nameLocation, bool value)
{
}

void DirectXShaderProgram::setInt(int nameLocation, int value)
{
}

void DirectXShaderProgram::setFloat(int nameLocation, float value)
{
}

void DirectXShaderProgram::setColor(int nameLocation, const Color &color)
{
}

void DirectXShaderProgram::setColor32(int nameLocation, const Color32 &color)
{
}

void DirectXShaderProgram::setVec2(int nameLocation, const glm::vec2 &vec)
{
}

void DirectXShaderProgram::setVec3(int nameLocation, const glm::vec3 &vec)
{
}

void DirectXShaderProgram::setVec4(int nameLocation, const glm::vec4 &vec)
{
}

void DirectXShaderProgram::setMat2(int nameLocation, const glm::mat2 &mat)
{
}

void DirectXShaderProgram::setMat3(int nameLocation, const glm::mat3 &mat)
{
}

void DirectXShaderProgram::setMat4(int nameLocation, const glm::mat4 &mat)
{
}

void DirectXShaderProgram::setTexture2D(int nameLocation, int texUnit, TextureHandle* tex)
{
}

void DirectXShaderProgram::setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{
}

bool DirectXShaderProgram::getBool(const char *name) const
{
    return false;
}

int DirectXShaderProgram::getInt(const char *name) const
{
    return -1;
}

float DirectXShaderProgram::getFloat(const char *name) const
{
    return 0.0f;
}

Color DirectXShaderProgram::getColor(const char *name) const
{
    return Color::black;
}

glm::vec2 DirectXShaderProgram::getVec2(const char *name) const
{
    return glm::vec2();
}

glm::vec3 DirectXShaderProgram::getVec3(const char *name) const
{
    return glm::vec3();
}

glm::vec4 DirectXShaderProgram::getVec4(const char *name) const
{
    return glm::vec4();
}

glm::mat2 DirectXShaderProgram::getMat2(const char *name) const
{
    return glm::mat2();
}

glm::mat3 DirectXShaderProgram::getMat3(const char *name) const
{
    return glm::mat3();
}

glm::mat4 DirectXShaderProgram::getMat4(const char *name) const
{
    return glm::mat4();
}

int DirectXShaderProgram::getTexture2D(const char *name, int texUnit) const
{
    return -1;
}

bool DirectXShaderProgram::getBool(int nameLocation) const 
{
    return false;
}

int DirectXShaderProgram::getInt(int nameLocation) const 
{
    return -1;
}

float DirectXShaderProgram::getFloat(int nameLocation) const 
{
    return 0.0f;
}

Color DirectXShaderProgram::getColor(int nameLocation) const 
{
    return Color::black;
}

Color32 DirectXShaderProgram::getColor32(int nameLocation) const 
{
    return Color32::black;
}

glm::vec2 DirectXShaderProgram::getVec2(int nameLocation) const
{
    return glm::vec2();
}

glm::vec3 DirectXShaderProgram::getVec3(int nameLocation) const
{
    return glm::vec3();
}

glm::vec4 DirectXShaderProgram::getVec4(int nameLocation) const
{
    return glm::vec4();
}

glm::mat2 DirectXShaderProgram::getMat2(int nameLocation) const
{
    return glm::mat2();
}

glm::mat3 DirectXShaderProgram::getMat3(int nameLocation) const
{
    return glm::mat3();
}

glm::mat4 DirectXShaderProgram::getMat4(int nameLocation) const
{
    return glm::mat4();
}

int DirectXShaderProgram::getTexture2D(int nameLocation, int texUnit) const
{
    return -1;
}

void *DirectXShaderProgram::getHandle()
{
    return nullptr;
}