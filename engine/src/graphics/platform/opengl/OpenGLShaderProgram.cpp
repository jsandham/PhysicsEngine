#include "../../../../include/graphics/platform/opengl/OpenGLShaderProgram.h"
#include "../../../../include/graphics/Renderer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include "../../../../include/core/Log.h"
#include "../../../../include/core/Shader.h"

#include <GL/glew.h>
#include <array>
#include <iostream>

using namespace PhysicsEngine;

OpenGLShaderProgram::OpenGLShaderProgram()
{
    mHandle = glCreateProgram();

    assert(mHandle > 0);
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

    memset(mStatus.mVertexCompileLog, 0, sizeof(mStatus.mVertexCompileLog));
    memset(mStatus.mFragmentCompileLog, 0, sizeof(mStatus.mFragmentCompileLog));
    memset(mStatus.mGeometryCompileLog, 0, sizeof(mStatus.mGeometryCompileLog));
    memset(mStatus.mLinkLog, 0, sizeof(mStatus.mLinkLog));
}

void OpenGLShaderProgram::load(const std::string &name, const std::string &vertex, const std::string &fragment)
{
    this->load(name, vertex, fragment, "");
}

struct GLUniform
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

struct GLAttribute
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

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

    // for (int j = 0; j < numBlocks; j++)
    //{
    //     GLint nameLen;
    //     CHECK_ERROR(glGetActiveUniformBlockiv(mHandle, j, GL_UNIFORM_BLOCK_NAME_LENGTH, &nameLen));

    //    std::vector<GLchar> name;
    //    name.resize(nameLen);
    //    CHECK_ERROR(glGetActiveUniformBlockName(mHandle, j, nameLen, NULL, &name[0]));

    //    std::string blockName;
    //    blockName.assign(name.begin(), name.end() - 1);

    //    std::cout << "blockName: " << blockName << std::endl;
    //    //nameList.push_back(std::string());
    //    //nameList.back().assign(name.begin(), name.end() - 1); // Remove the null terminator.
    //}

    // Uniform buffers
    GLint numBlocks;
    CHECK_ERROR(glGetProgramiv(mHandle, GL_ACTIVE_UNIFORM_BLOCKS, &numBlocks));

    std::array<GLenum, 2> blockProperties{GL_NAME_LENGTH, GL_NUM_ACTIVE_VARIABLES};
    std::array<GLint, 2> blockData{};

    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx)
    {
        CHECK_ERROR(glGetProgramResourceiv(mHandle, GL_UNIFORM_BLOCK, blockIdx, (GLsizei)blockProperties.size(),
                                           blockProperties.data(), (GLsizei)blockData.size(), nullptr,
                                           blockData.data()));

        // Retrieve name
        std::vector<char> blockName(blockData[0]);
        CHECK_ERROR(glGetProgramResourceName(mHandle, GL_UNIFORM_BLOCK, blockIdx, (GLsizei)blockName.size() + 1,
                                             nullptr, blockName.data()));

        // Retrieve indices of uniforms that are a member of this block.
        std::vector<GLint> uniformIdxs(blockData[1]);

        GLenum member = GL_ACTIVE_VARIABLES;
        CHECK_ERROR(glGetProgramResourceiv(mHandle, GL_UNIFORM_BLOCK, blockIdx, 1, &member, (GLsizei)uniformIdxs.size(),
                                           nullptr, uniformIdxs.data()));

        //std::cout << "blockName: " << std::string(blockName.data()) << " uniform count: " << blockData[1] << std::endl;

        std::array<GLenum, 2> uniformProperties{GL_NAME_LENGTH, GL_TYPE};
        std::array<GLint, 2> uniformData{};
        for (int uniformIdx = 0; uniformIdx < blockData[1]; uniformIdx++)
        {
            CHECK_ERROR(glGetProgramResourceiv(mHandle, GL_UNIFORM, uniformIdx, (GLsizei)uniformProperties.size(),
                                               uniformProperties.data(), (GLsizei)blockData.size(), nullptr,
                                               uniformData.data()));

            std::vector<char> uniformName(uniformData[0]);
            CHECK_ERROR(glGetProgramResourceName(mHandle, GL_UNIFORM, uniformIdx, (GLsizei)uniformName.size() + 1,
                                                 nullptr, uniformName.data()));

            //std::cout << "uniform name: " << std::string(uniformName.data()) << " type: " << uniformData[1]
            //          << std::endl;

            // mUniforms[].mName = std::string(uniformName.data());
            // mUniforms[].mBufferName = std::string(blockName.data());
            // mUniforms[].mType =
        }
    }

    // Standalone Uniforms
    GLint count;
    CHECK_ERROR(glGetProgramiv(mHandle, GL_ACTIVE_UNIFORMS, &count));

    mUniforms.resize(count);
    mMaterialUniforms.resize(count);
    mLocations.resize(count);
    mUniformIds.resize(count);

    int uniformCount = 0;
    int materialUniformCount = 0;
    for (size_t uniformIdx = 0; uniformIdx < mUniforms.size(); uniformIdx++)
    {
        GLUniform uniform;
        CHECK_ERROR(glGetActiveUniform(mHandle, (GLuint)uniformIdx, 32, &uniform.nameLength, &uniform.size,
                                       &uniform.type, &uniform.name[0]));

        int loc = findUniformLocation(&uniform.name[0]);

        if (loc >= 0)
        {
            mUniforms[uniformCount].mBufferName = "$Global";
            mUniforms[uniformCount].mName = std::string(uniform.name);
            switch (uniform.type)
            {
            case GL_INT:
                mUniforms[uniformCount].mType = ShaderUniformType::Int;
                break;
            case GL_FLOAT:
                mUniforms[uniformCount].mType = ShaderUniformType::Float;
                break;
            case GL_FLOAT_VEC2:
                mUniforms[uniformCount].mType = ShaderUniformType::Vec2;
                break;
            case GL_FLOAT_VEC3:
                mUniforms[uniformCount].mType = ShaderUniformType::Vec3;
                break;
            case GL_FLOAT_VEC4:
                mUniforms[uniformCount].mType = ShaderUniformType::Vec4;
                break;
            case GL_FLOAT_MAT2:
                mUniforms[uniformCount].mType = ShaderUniformType::Mat2;
                break;
            case GL_FLOAT_MAT3:
                mUniforms[uniformCount].mType = ShaderUniformType::Mat3;
                break;
            case GL_FLOAT_MAT4:
                mUniforms[uniformCount].mType = ShaderUniformType::Mat4;
                break;
            case GL_SAMPLER_2D:
                mUniforms[uniformCount].mType = ShaderUniformType::Sampler2D;
                break;
            case GL_SAMPLER_CUBE:
                mUniforms[uniformCount].mType = ShaderUniformType::SamplerCube;
                break;
            }

            mUniforms[uniformCount].mUniformId = Shader::uniformToId(&uniform.name[0]);
            mUniforms[uniformCount].mTex = nullptr;
            memset(mUniforms[uniformCount].mData, '\0', 64);

            mLocations[uniformCount] = loc;
            mUniformIds[uniformCount] = mUniforms[uniformCount].mUniformId;

            if (mUniforms[uniformCount].mName.find("material") != std::string::npos)
            {
                mMaterialUniforms[materialUniformCount] = mUniforms[uniformCount];
                materialUniformCount++;
            }

            uniformCount++;
        }
    }

    mUniforms.resize(uniformCount);
    mMaterialUniforms.resize(materialUniformCount);
    mLocations.resize(uniformCount);
    mUniformIds.resize(uniformCount);

    // Attributes
    GLint attributeCount;
    CHECK_ERROR(glGetProgramiv(mHandle, GL_ACTIVE_ATTRIBUTES, &attributeCount));

    mAttributes.resize(attributeCount);

    for (int j = 0; j < attributeCount; j++)
    {
        GLAttribute attrib;
        CHECK_ERROR(
            glGetActiveAttrib(mHandle, (GLuint)j, 32, &attrib.nameLength, &attrib.size, &attrib.type, &attrib.name[0]));

        mAttributes[j].mName = std::string(attrib.name);
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

std::vector<ShaderUniform> OpenGLShaderProgram::getUniforms() const
{
    return mUniforms;
}

std::vector<ShaderUniform> OpenGLShaderProgram::getMaterialUniforms() const
{
    return mMaterialUniforms;
}

std::vector<ShaderAttribute> OpenGLShaderProgram::getAttributes() const
{
    return mAttributes;
}

void OpenGLShaderProgram::setBool(const char *name, bool value)
{
    // TODO: Is it faster to call this->setBool(Shader::uniformToId(name), value);??
    CHECK_ERROR(glUniform1i(findUniformLocation(name), (int)value));
}

void OpenGLShaderProgram::setInt(const char *name, int value)
{
    CHECK_ERROR(glUniform1i(findUniformLocation(name), value));
}

void OpenGLShaderProgram::setFloat(const char *name, float value)
{
    CHECK_ERROR(glUniform1f(findUniformLocation(name), value));
}

void OpenGLShaderProgram::setColor(const char *name, const Color &color)
{
    CHECK_ERROR(glUniform4fv(findUniformLocation(name), 1, static_cast<const GLfloat *>(&color.mR)));
}

void OpenGLShaderProgram::setColor32(const char *name, const Color32 &color)
{
    CHECK_ERROR(glUniform4ui(findUniformLocation(name), static_cast<GLuint>(color.mR), static_cast<GLuint>(color.mG),
                             static_cast<GLuint>(color.mB), static_cast<GLuint>(color.mA)));
}

void OpenGLShaderProgram::setVec2(const char *name, const glm::vec2 &vec)
{
    CHECK_ERROR(glUniform2fv(findUniformLocation(name), 1, &vec[0]));
}

void OpenGLShaderProgram::setVec3(const char *name, const glm::vec3 &vec)
{
    CHECK_ERROR(glUniform3fv(findUniformLocation(name), 1, &vec[0]));
}

void OpenGLShaderProgram::setVec4(const char *name, const glm::vec4 &vec)
{
    CHECK_ERROR(glUniform4fv(findUniformLocation(name), 1, &vec[0]));
}

void OpenGLShaderProgram::setMat2(const char *name, const glm::mat2 &mat)
{
    CHECK_ERROR(glUniformMatrix2fv(findUniformLocation(name), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat3(const char *name, const glm::mat3 &mat)
{
    CHECK_ERROR(glUniformMatrix3fv(findUniformLocation(name), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat4(const char *name, const glm::mat4 &mat)
{
    CHECK_ERROR(glUniformMatrix4fv(findUniformLocation(name), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setTexture2D(const char *name, int texUnit, void *tex)
{
    CHECK_ERROR(glUniform1i(findUniformLocation(name), texUnit));

    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
    if (tex != nullptr)
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int *>(tex)));
    }
    else
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    }
}

void OpenGLShaderProgram::setTexture2Ds(const char *name, const std::vector<int> &texUnits, int count,
                                        const std::vector<void *> &texs)
{
    CHECK_ERROR(glUniform1iv(findUniformLocation(name), count, texUnits.data()));

    for (int i = 0; i < count; i++)
    {
        CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnits[i]));
        if (texs[i] != nullptr)
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int *>(texs[i])));
        }
        else
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
        }
    }
}

void OpenGLShaderProgram::setBool(int uniformId, bool value)
{
    CHECK_ERROR(glUniform1i(findUniformLocation(uniformId), (int)value));
}

void OpenGLShaderProgram::setInt(int uniformId, int value)
{
    CHECK_ERROR(glUniform1i(findUniformLocation(uniformId), value));
}

void OpenGLShaderProgram::setFloat(int uniformId, float value)
{
    CHECK_ERROR(glUniform1f(findUniformLocation(uniformId), value));
}

void OpenGLShaderProgram::setColor(int uniformId, const Color &color)
{
    CHECK_ERROR(glUniform4fv(findUniformLocation(uniformId), 1, static_cast<const GLfloat *>(&color.mR)));
}

void OpenGLShaderProgram::setColor32(int uniformId, const Color32 &color)
{
    CHECK_ERROR(glUniform4ui(findUniformLocation(uniformId), static_cast<GLuint>(color.mR),
                             static_cast<GLuint>(color.mG), static_cast<GLuint>(color.mB),
                             static_cast<GLuint>(color.mA)));
}

void OpenGLShaderProgram::setVec2(int uniformId, const glm::vec2 &vec)
{
    CHECK_ERROR(glUniform2fv(findUniformLocation(uniformId), 1, &vec[0]));
}

void OpenGLShaderProgram::setVec3(int uniformId, const glm::vec3 &vec)
{
    CHECK_ERROR(glUniform3fv(findUniformLocation(uniformId), 1, &vec[0]));
}

void OpenGLShaderProgram::setVec4(int uniformId, const glm::vec4 &vec)
{
    CHECK_ERROR(glUniform4fv(findUniformLocation(uniformId), 1, &vec[0]));
}

void OpenGLShaderProgram::setMat2(int uniformId, const glm::mat2 &mat)
{
    CHECK_ERROR(glUniformMatrix2fv(findUniformLocation(uniformId), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat3(int uniformId, const glm::mat3 &mat)
{
    CHECK_ERROR(glUniformMatrix3fv(findUniformLocation(uniformId), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setMat4(int uniformId, const glm::mat4 &mat)
{
    CHECK_ERROR(glUniformMatrix4fv(findUniformLocation(uniformId), 1, GL_FALSE, &mat[0][0]));
}

void OpenGLShaderProgram::setTexture2D(int uniformId, int texUnit, void *tex)
{
    CHECK_ERROR(glUniform1i(findUniformLocation(uniformId), texUnit));

    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
    if (tex != nullptr)
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int *>(tex)));
    }
    else
    {
        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    }
}

void OpenGLShaderProgram::setTexture2Ds(int uniformId, const std::vector<int> &texUnits, int count,
                                        const std::vector<void *> &texs)
{
    CHECK_ERROR(glUniform1iv(findUniformLocation(uniformId), count, texUnits.data()));

    for (int i = 0; i < count; i++)
    {
        CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnits[i]));
        if (texs[i] != nullptr)
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int *>(texs[i])));
        }
        else
        {
            CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
        }
    }
}

bool OpenGLShaderProgram::getBool(const char *name) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, findUniformLocation(name), &value));

    return (bool)value;
}

int OpenGLShaderProgram::getInt(const char *name) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, findUniformLocation(name), &value));

    return value;
}

float OpenGLShaderProgram::getFloat(const char *name) const
{
    float value = 0.0f;
    CHECK_ERROR(glGetUniformfv(mHandle, findUniformLocation(name), &value));

    return value;
}

Color OpenGLShaderProgram::getColor(const char *name) const
{
    Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(Color), &color.mR));

    return color;
}

Color32 OpenGLShaderProgram::getColor32(const char *name) const
{
    Color32 color = Color32(0, 0, 0, 255);

    GLuint c[4];
    CHECK_ERROR(glGetnUniformuiv(mHandle, findUniformLocation(name), 4 * sizeof(GLuint), &c[0]));

    color.mR = static_cast<unsigned char>(c[0]);
    color.mG = static_cast<unsigned char>(c[1]);
    color.mB = static_cast<unsigned char>(c[2]);
    color.mA = static_cast<unsigned char>(c[3]);

    return color;
}

glm::vec2 OpenGLShaderProgram::getVec2(const char *name) const
{
    glm::vec2 value = glm::vec2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::vec2), &value[0]));

    return value;
}

glm::vec3 OpenGLShaderProgram::getVec3(const char *name) const
{
    glm::vec3 value = glm::vec3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::vec3), &value[0]));

    return value;
}

glm::vec4 OpenGLShaderProgram::getVec4(const char *name) const
{
    glm::vec4 value = glm::vec4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::vec4), &value[0]));

    return value;
}

glm::mat2 OpenGLShaderProgram::getMat2(const char *name) const
{
    glm::mat2 value = glm::mat2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::mat2), &value[0][0]));

    return value;
}

glm::mat3 OpenGLShaderProgram::getMat3(const char *name) const
{
    glm::mat3 value = glm::mat3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::mat3), &value[0][0]));

    return value;
}

glm::mat4 OpenGLShaderProgram::getMat4(const char *name) const
{
    glm::mat4 value = glm::mat4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(name), sizeof(glm::mat4), &value[0][0]));

    return value;
}

bool OpenGLShaderProgram::getBool(int uniformId) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, findUniformLocation(uniformId), &value));

    return (bool)value;
}

int OpenGLShaderProgram::getInt(int uniformId) const
{
    int value = 0;
    CHECK_ERROR(glGetUniformiv(mHandle, findUniformLocation(uniformId), &value));

    return value;
}

float OpenGLShaderProgram::getFloat(int uniformId) const
{
    float value = 0.0f;
    CHECK_ERROR(glGetUniformfv(mHandle, findUniformLocation(uniformId), &value));

    return value;
}

Color OpenGLShaderProgram::getColor(int uniformId) const
{
    Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(Color), &color.mR));

    return color;
}

Color32 OpenGLShaderProgram::getColor32(int uniformId) const
{
    Color32 color = Color32(0, 0, 0, 255);

    GLuint c[4];
    CHECK_ERROR(glGetnUniformuiv(mHandle, findUniformLocation(uniformId), 4 * sizeof(GLuint), &c[0]));

    color.mR = static_cast<unsigned char>(c[0]);
    color.mG = static_cast<unsigned char>(c[1]);
    color.mB = static_cast<unsigned char>(c[2]);
    color.mA = static_cast<unsigned char>(c[3]);

    return color;
}

glm::vec2 OpenGLShaderProgram::getVec2(int uniformId) const
{
    glm::vec2 value = glm::vec2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::vec2), &value[0]));

    return value;
}

glm::vec3 OpenGLShaderProgram::getVec3(int uniformId) const
{
    glm::vec3 value = glm::vec3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::vec3), &value[0]));

    return value;
}

glm::vec4 OpenGLShaderProgram::getVec4(int uniformId) const
{
    glm::vec4 value = glm::vec4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::vec4), &value[0]));

    return value;
}

glm::mat2 OpenGLShaderProgram::getMat2(int uniformId) const
{
    glm::mat2 value = glm::mat2(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::mat2), &value[0][0]));

    return value;
}

glm::mat3 OpenGLShaderProgram::getMat3(int uniformId) const
{
    glm::mat3 value = glm::mat3(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::mat3), &value[0][0]));

    return value;
}

glm::mat4 OpenGLShaderProgram::getMat4(int uniformId) const
{
    glm::mat4 value = glm::mat4(0.0f);
    CHECK_ERROR(glGetnUniformfv(mHandle, findUniformLocation(uniformId), sizeof(glm::mat4), &value[0][0]));

    return value;
}

int OpenGLShaderProgram::findUniformLocation(const char *name) const
{
    // Returns -1 if uniform is part of uniform block or name does not correspond to a uniform
    return glGetUniformLocation(mHandle, name);
}

int OpenGLShaderProgram::findUniformLocation(int uniformId) const
{
    for (size_t i = 0; i < mUniformIds.size(); i++)
    {
        if (uniformId == mUniformIds[i])
        {
            return mLocations[i];
        }
    }

    return -1;
}