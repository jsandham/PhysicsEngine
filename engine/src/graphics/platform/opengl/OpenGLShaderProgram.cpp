#include "../../../../include/graphics/platform/opengl/OpenGLShaderProgram.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/graphics/Renderer.h"

#include "../../../../include/core/Log.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLShaderProgram::OpenGLShaderProgram()
{
    mHandle = glCreateProgram();
}

OpenGLShaderProgram::~OpenGLShaderProgram()
{
    CHECK_ERROR(glDeleteProgram(mHandle));
}

void OpenGLShaderProgram::load(const std::string &name, const std::string &vertex, const std::string &fragment,
                               const std::string &geometry)
{
    mName = name;
    mVertex = vertex;
    mFragment = fragment;
    mGeometry = geometry;
}

void OpenGLShaderProgram::load(const std::string &name, const std::string &vertex, const std::string &fragment)
{
    this->load(name, vertex, fragment, "");
}

void OpenGLShaderProgram::compile()
{
    memset(mStatus.mVertexCompileLog, 0, sizeof(mStatus.mVertexCompileLog));
    memset(mStatus.mFragmentCompileLog, 0, sizeof(mStatus.mFragmentCompileLog));
    memset(mStatus.mGeometryCompileLog, 0, sizeof(mStatus.mGeometryCompileLog));
    memset(mStatus.mLinkLog, 0, sizeof(mStatus.mLinkLog));

    const GLchar *vertexShaderCharPtr = mVertex.c_str();
    const GLchar *geometryShaderCharPtr = mGeometry.c_str();
    const GLchar *fragmentShaderCharPtr = mFragment.c_str();

    // Compile vertex shader
    GLuint vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
    CHECK_ERROR(glShaderSource(vertexShaderObj, 1, &vertexShaderCharPtr, NULL));
    CHECK_ERROR(glCompileShader(vertexShaderObj));
    CHECK_ERROR(glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &mStatus.mVertexShaderCompiled));
    if (!mStatus.mVertexShaderCompiled)
    {
        CHECK_ERROR(glGetShaderInfoLog(vertexShaderObj, 512, NULL, mStatus.mVertexCompileLog));

        std::string message = "Shader: Vertex shader compilation failed (" + mName + ")\n";
        Log::error(message.c_str());
    }

    // Compile fragment shader
    GLuint fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    CHECK_ERROR(glShaderSource(fragmentShaderObj, 1, &fragmentShaderCharPtr, NULL));
    CHECK_ERROR(glCompileShader(fragmentShaderObj));
    CHECK_ERROR(glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &mStatus.mFragmentShaderCompiled));
    if (!mStatus.mFragmentShaderCompiled)
    {
        CHECK_ERROR(glGetShaderInfoLog(fragmentShaderObj, 512, NULL, mStatus.mFragmentCompileLog));

        std::string message = "Shader: Fragment shader compilation failed (" + mName + ")\n";
        Log::error(message.c_str());
    }

    // Compile geometry shader
    GLuint geometryShaderObj = 0;
    if (!mGeometry.empty())
    {
        geometryShaderObj = glCreateShader(GL_GEOMETRY_SHADER);
        CHECK_ERROR(glShaderSource(geometryShaderObj, 1, &geometryShaderCharPtr, NULL));
        CHECK_ERROR(glCompileShader(geometryShaderObj));
        CHECK_ERROR(glGetShaderiv(geometryShaderObj, GL_COMPILE_STATUS, &mStatus.mGeometryShaderCompiled));
        if (!mStatus.mGeometryShaderCompiled)
        {
            CHECK_ERROR(glGetShaderInfoLog(geometryShaderObj, 512, NULL, mStatus.mGeometryCompileLog));

            std::string message = "Shader: Geometry shader compilation failed (" + mName + ")\n";
            Log::error(message.c_str());
        }
    }

    // Attach shader objects to shader program
    CHECK_ERROR(glAttachShader(mHandle, vertexShaderObj));
    CHECK_ERROR(glAttachShader(mHandle, fragmentShaderObj));
    if (geometryShaderObj != 0)
    {
        CHECK_ERROR(glAttachShader(mHandle, geometryShaderObj));
    }

    // Link shader program
    CHECK_ERROR(glLinkProgram(mHandle));
    CHECK_ERROR(glGetProgramiv(mHandle, GL_LINK_STATUS, &mStatus.mShaderLinked));
    if (!mStatus.mShaderLinked)
    {
        CHECK_ERROR(glGetProgramInfoLog(mHandle, 512, NULL, mStatus.mLinkLog));

        std::string message = "Shader: " + mName + " program linking failed\n";
        Log::error(message.c_str());
    }

    // Detach shader objects from shader program
    CHECK_ERROR(glDetachShader(mHandle, vertexShaderObj));
    CHECK_ERROR(glDetachShader(mHandle, fragmentShaderObj));
    if (geometryShaderObj != 0)
    {
        CHECK_ERROR(glDetachShader(mHandle, geometryShaderObj));
    }

    // Delete shader objects
    CHECK_ERROR(glDeleteShader(vertexShaderObj));
    CHECK_ERROR(glDeleteShader(fragmentShaderObj));
    if (!mGeometry.empty())
    {
        CHECK_ERROR(glDeleteShader(geometryShaderObj));
    }

    GLuint blockIndex = glGetUniformBlockIndex(mHandle, "CameraBlock");
    if (blockIndex != GL_INVALID_INDEX)
    {
        CHECK_ERROR(glUniformBlockBinding(mHandle, blockIndex, 0));
    }

    blockIndex = glGetUniformBlockIndex(mHandle, "LightBlock");
    if (blockIndex != GL_INVALID_INDEX)
    {
        CHECK_ERROR(glUniformBlockBinding(mHandle, blockIndex, 1));
    }
}

void OpenGLShaderProgram::bind()
{
    CHECK_ERROR(glUseProgram(mHandle));
}

void OpenGLShaderProgram::unbind()
{
    CHECK_ERROR(glUseProgram(0));
}

int OpenGLShaderProgram::findUniformLocation(const std::string &name) const
{
    return glGetUniformLocation(mHandle, name.c_str());
}

struct Uniform
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

struct Attribute
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

std::vector<ShaderUniform> OpenGLShaderProgram::getUniforms() const
{
    GLint uniformCount;
    CHECK_ERROR(glGetProgramiv(mHandle, GL_ACTIVE_UNIFORMS, &uniformCount));

    std::vector<ShaderUniform> uniforms(uniformCount);

    for (size_t j = 0; j < uniforms.size(); j++)
    {
        Uniform uniform;
        CHECK_ERROR(glGetActiveUniform(mHandle, (GLuint)j, 32, &uniform.nameLength, &uniform.size, &uniform.type,
                                       &uniform.name[0]));

        uniforms[j].mName = std::string(uniform.name);
        switch (uniform.type)
        {
        case GL_INT:
            uniforms[j].mType = ShaderUniformType::Int;
            break;
        case GL_FLOAT:
            uniforms[j].mType = ShaderUniformType::Float;
            break;
        case GL_FLOAT_VEC2:
            uniforms[j].mType = ShaderUniformType::Vec2;
            break;
        case GL_FLOAT_VEC3:
            uniforms[j].mType = ShaderUniformType::Vec3;
            break;
        case GL_FLOAT_VEC4:
            uniforms[j].mType = ShaderUniformType::Vec4;
            break;
        case GL_FLOAT_MAT2:
            uniforms[j].mType = ShaderUniformType::Mat2;
            break;
        case GL_FLOAT_MAT3:
            uniforms[j].mType = ShaderUniformType::Mat3;
            break;
        case GL_FLOAT_MAT4:
            uniforms[j].mType = ShaderUniformType::Mat4;
            break;
        case GL_SAMPLER_2D:
            uniforms[j].mType = ShaderUniformType::Sampler2D;
            break;
        case GL_SAMPLER_CUBE:
            uniforms[j].mType = ShaderUniformType::SamplerCube;
            break;
        }

        uniforms[j].mUniformId = 0;
        uniforms[j].mTex = nullptr;
        memset(uniforms[j].mData, '\0', 64);
    }

    return uniforms;
}

std::vector<ShaderAttribute> OpenGLShaderProgram::getAttributes() const
{
    GLint attributeCount;
    CHECK_ERROR(glGetProgramiv(mHandle, GL_ACTIVE_ATTRIBUTES, &attributeCount));

    std::vector<ShaderAttribute> attributes(attributeCount);

    for (int j = 0; j < attributeCount; j++)
    {
        Attribute attrib;
        CHECK_ERROR(
            glGetActiveAttrib(mHandle, (GLuint)j, 32, &attrib.nameLength, &attrib.size, &attrib.type, &attrib.name[0]));

        attributes[j].mName = std::string(attrib.name);
    }

    return attributes;
}

void OpenGLShaderProgram::setBool(const char *name, bool value)
{
    this->setBool(glGetUniformLocation(mHandle, name), value);
}

void OpenGLShaderProgram::setInt(const char *name, int value)
{
    this->setInt(glGetUniformLocation(mHandle, name), value);
}

void OpenGLShaderProgram::setFloat(const char *name, float value)
{
    this->setFloat(glGetUniformLocation(mHandle, name), value);
}

void OpenGLShaderProgram::setColor(const char *name, const Color &color)
{
    this->setColor(glGetUniformLocation(mHandle, name), color);
}

void OpenGLShaderProgram::setColor32(const char *name, const Color32 &color)
{
    this->setColor32(glGetUniformLocation(mHandle, name), color);
}

void OpenGLShaderProgram::setVec2(const char *name, const glm::vec2 &vec)
{
    this->setVec2(glGetUniformLocation(mHandle, name), vec);
}

void OpenGLShaderProgram::setVec3(const char *name, const glm::vec3 &vec)
{
    this->setVec3(glGetUniformLocation(mHandle, name), vec);
}

void OpenGLShaderProgram::setVec4(const char *name, const glm::vec4 &vec)
{
    this->setVec4(glGetUniformLocation(mHandle, name), vec);
}

void OpenGLShaderProgram::setMat2(const char *name, const glm::mat2 &mat)
{
    this->setMat2(glGetUniformLocation(mHandle, name), mat);
}

void OpenGLShaderProgram::setMat3(const char *name, const glm::mat3 &mat)
{
    this->setMat3(glGetUniformLocation(mHandle, name), mat);
}

void OpenGLShaderProgram::setMat4(const char *name, const glm::mat4 &mat)
{
    this->setMat4(glGetUniformLocation(mHandle, name), mat);
}

void OpenGLShaderProgram::setTexture2D(const char *name, int texUnit, TextureHandle* tex)
{
    this->setTexture2D(glGetUniformLocation(mHandle, name), texUnit, tex);
}

void OpenGLShaderProgram::setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{
    this->setTexture2Ds(glGetUniformLocation(mHandle, name), texUnits, count, texs);
}

void OpenGLShaderProgram::setBool(int nameLocation, bool value)
{
    CHECK_ERROR(glUniform1i(nameLocation, (int)value));
}

void OpenGLShaderProgram::setInt(int nameLocation, int value)
{
    CHECK_ERROR(glUniform1i(nameLocation, value));
}

void OpenGLShaderProgram::setFloat(int nameLocation, float value)
{
    CHECK_ERROR(glUniform1f(nameLocation, value));
}

void OpenGLShaderProgram::setColor(int nameLocation, const Color &color)
{
    CHECK_ERROR(glUniform4fv(nameLocation, 1, static_cast<const GLfloat *>(&color.mR)));
}

void OpenGLShaderProgram::setColor32(int nameLocation, const Color32 &color)
{
    CHECK_ERROR(glUniform4ui(nameLocation, static_cast<GLuint>(color.mR), static_cast<GLuint>(color.mG),
                             static_cast<GLuint>(color.mB), static_cast<GLuint>(color.mA)));
}

void OpenGLShaderProgram::setVec2(int nameLocation, const glm::vec2 &vec)
{
    CHECK_ERROR(glUniform2fv(nameLocation, 1, &vec[0]));
}

void OpenGLShaderProgram::setVec3(int nameLocation, const glm::vec3 &vec)
{
    CHECK_ERROR(glUniform3fv(nameLocation, 1, &vec[0]));
}

void OpenGLShaderProgram::setVec4(int nameLocation, const glm::vec4 &vec)
{
    CHECK_ERROR(glUniform4fv(nameLocation, 1, &vec[0]));
}

void OpenGLShaderProgram::setMat2(int nameLocation, const glm::mat2 &mat)
{
    CHECK_ERROR(glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat3(int nameLocation, const glm::mat3 &mat)
{
    CHECK_ERROR(glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat4(int nameLocation, const glm::mat4 &mat)
{
    CHECK_ERROR(glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setTexture2D(int nameLocation, int texUnit, TextureHandle* tex)
{
    CHECK_ERROR(glUniform1i(nameLocation, texUnit));

    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
    if (tex != nullptr)
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int*>(tex->getHandle())));
    }
    else
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    }
}

void OpenGLShaderProgram::setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{
    CHECK_ERROR(glUniform1iv(nameLocation, count, texUnits.data()));

    for (int i = 0; i < count; i++)
    {
        CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnits[i]));
        if (texs[i] != nullptr)
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int *>(texs[i]->getHandle())));           
        }
        else
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
        }
    }
}

bool OpenGLShaderProgram::getBool(const char *name) const
{
    return this->getBool(glGetUniformLocation(mHandle, name));
}

int OpenGLShaderProgram::getInt(const char *name) const
{
    return this->getInt(glGetUniformLocation(mHandle, name));
}

float OpenGLShaderProgram::getFloat(const char *name) const
{
    return this->getFloat(glGetUniformLocation(mHandle, name));
}

Color OpenGLShaderProgram::getColor(const char *name) const
{
    return this->getColor(glGetUniformLocation(mHandle, name));
}

glm::vec2 OpenGLShaderProgram::getVec2(const char *name) const
{
    return this->getVec2(glGetUniformLocation(mHandle, name));
}

glm::vec3 OpenGLShaderProgram::getVec3(const char *name) const
{
    return this->getVec3(glGetUniformLocation(mHandle, name));
}

glm::vec4 OpenGLShaderProgram::getVec4(const char *name) const
{
    return this->getVec4(glGetUniformLocation(mHandle, name));
}

glm::mat2 OpenGLShaderProgram::getMat2(const char *name) const
{
    return this->getMat2(glGetUniformLocation(mHandle, name));
}

glm::mat3 OpenGLShaderProgram::getMat3(const char *name) const
{
    return this->getMat3(glGetUniformLocation(mHandle, name));
}

glm::mat4 OpenGLShaderProgram::getMat4(const char *name) const
{
    return this->getMat4(glGetUniformLocation(mHandle, name));
}

bool OpenGLShaderProgram::getBool(int nameLocation) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, nameLocation, &value));

    return (bool)value;
}

int OpenGLShaderProgram::getInt(int nameLocation) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, nameLocation, &value));

    return value;
}

float OpenGLShaderProgram::getFloat(int nameLocation) const
{
    float value = 0.0f;
    CHECK_ERROR(glGetUniformfv(mHandle, nameLocation, &value));

    return value;
}

Color OpenGLShaderProgram::getColor(int nameLocation) const
{
    Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(Color), &color.mR));

    return color;
}

Color32 OpenGLShaderProgram::getColor32(int nameLocation) const
{
    Color32 color = Color32(0, 0, 0, 255);

    GLuint c[4];
    CHECK_ERROR(glGetnUniformuiv(mHandle, nameLocation, 4 * sizeof(GLuint), &c[0]));

    color.mR = static_cast<unsigned char>(c[0]);
    color.mG = static_cast<unsigned char>(c[1]);
    color.mB = static_cast<unsigned char>(c[2]);
    color.mA = static_cast<unsigned char>(c[3]);

    return color;
}

glm::vec2 OpenGLShaderProgram::getVec2(int nameLocation) const
{
    glm::vec2 value = glm::vec2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::vec2), &value[0]));

    return value;
}

glm::vec3 OpenGLShaderProgram::getVec3(int nameLocation) const
{
    glm::vec3 value = glm::vec3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::vec3), &value[0]));

    return value;
}

glm::vec4 OpenGLShaderProgram::getVec4(int nameLocation) const
{
    glm::vec4 value = glm::vec4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::vec4), &value[0]));

    return value;
}

glm::mat2 OpenGLShaderProgram::getMat2(int nameLocation) const
{
    glm::mat2 value = glm::mat2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::mat2), &value[0][0]));

    return value;
}

glm::mat3 OpenGLShaderProgram::getMat3(int nameLocation) const
{
    glm::mat3 value = glm::mat3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::mat3), &value[0][0]));

    return value;
}

glm::mat4 OpenGLShaderProgram::getMat4(int nameLocation) const
{
    glm::mat4 value = glm::mat4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, nameLocation, sizeof(glm::mat4), &value[0][0]));

    return value;
}