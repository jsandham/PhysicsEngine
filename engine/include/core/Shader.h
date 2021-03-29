#ifndef SHADER_H__
#define SHADER_H__

#define NOMINMAX

#include <string>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "../glm/glm.hpp"

#include "Asset.h"
#include "Color.h"
#include "Guid.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
enum class RenderQueue
{
    Opaque = 0,
    Transparent = 1
};

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
    ShaderVersion mVersion;
    int mVariant;
    GLuint mHandle;
    bool mCompiled;
};

struct ShaderUniform
{
    char mData[64];
    char mName[32];      // variable name in GLSL (including block name if applicable)
    char mShortName[32]; // variable name in GLSL (excluding block name if applicable)
    char mBlockName[32]; // block name (empty string if not part of block)
    size_t mNameLength;  // length of name
    size_t mSize;        // size of the uniform
    GLenum mType;        // type of the uniform (float, vec3 or mat4, etc)
    int mVariant;        // variant this uniform occurs in
    int mLocation;       // uniform location in shader program
    size_t mIndex;       // what index in array of uniforms we are at
};

struct ShaderAttribute
{
    char mName[32];
};

class Shader : public Asset
{
  private:
    std::string mVertexSource;
    std::string mFragmentSource;
    std::string mGeometrySource;
    std::string mVertexShader;
    std::string mFragmentShader;
    std::string mGeometryShader;

    bool mAllProgramsCompiled;
    int mActiveProgram;
    std::vector<ShaderProgram> mPrograms;
    std::vector<ShaderUniform> mUniforms;
    std::vector<ShaderAttribute> mAttributes;

  public:
    Shader();
    Shader(Guid id);
    ~Shader();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const std::string &vertFilepath, const std::string &fragFilepath, const std::string &geoFilepath);

    bool isCompiled() const;
    bool contains(int variant) const;
    void add(int variant);
    void remove(int variant);

    void compile();
    void use(int program);
    void unuse();
    void setVertexShader(const std::string &vertexShader);
    void setGeometryShader(const std::string &geometryShader);
    void setFragmentShader(const std::string &fragmentShader);
    void setUniformBlock(const std::string &blockName, int bindingPoint) const;
    int findUniformLocation(const std::string &name, int program) const;
    int getProgramFromVariant(int variant) const;
    int getActiveProgram() const;

    std::vector<ShaderProgram> getPrograms() const;
    std::vector<ShaderUniform> getUniforms() const;
    std::vector<ShaderAttribute> getAttributeNames() const;
    std::string getVertexShader() const;
    std::string getGeometryShader() const;
    std::string getFragmentShader() const;

    void setBool(const char *name, bool value) const;
    void setInt(const char *name, int value) const;
    void setFloat(const char *name, float value) const;
    void setColor(const char *name, const Color &color) const;
    void setVec2(const char *name, const glm::vec2 &vec) const;
    void setVec3(const char *name, const glm::vec3 &vec) const;
    void setVec4(const char *name, const glm::vec4 &vec) const;
    void setMat2(const char *name, const glm::mat2 &mat) const;
    void setMat3(const char *name, const glm::mat3 &mat) const;
    void setMat4(const char *name, const glm::mat4 &mat) const;
    void setTexture2D(const char *name, int texUnit, int tex) const;

    void setBool(int nameLocation, bool value) const;
    void setInt(int nameLocation, int value) const;
    void setFloat(int nameLocation, float value) const;
    void setColor(int nameLocation, const Color &color) const;
    void setVec2(int nameLocation, const glm::vec2 &vec) const;
    void setVec3(int nameLocation, const glm::vec3 &vec) const;
    void setVec4(int nameLocation, const glm::vec4 &vec) const;
    void setMat2(int nameLocation, const glm::mat2 &mat) const;
    void setMat3(int nameLocation, const glm::mat3 &mat) const;
    void setMat4(int nameLocation, const glm::mat4 &mat) const;
    void setTexture2D(int nameLocation, int texUnit, int tex) const;

    bool getBool(const char *name) const;
    int getInt(const char *name) const;
    float getFloat(const char *name) const;
    Color getColor(const char *name) const;
    glm::vec2 getVec2(const char *name) const;
    glm::vec3 getVec3(const char *name) const;
    glm::vec4 getVec4(const char *name) const;
    glm::mat2 getMat2(const char *name) const;
    glm::mat3 getMat3(const char *name) const;
    glm::mat4 getMat4(const char *name) const;
    int getTexture2D(const char *name, int texUnit) const;

    bool getBool(int nameLocation) const;
    int getInt(int nameLocation) const;
    float getFloat(int nameLocation) const;
    Color getColor(int nameLocation) const;
    glm::vec2 getVec2(int nameLocation) const;
    glm::vec3 getVec3(int nameLocation) const;
    glm::vec4 getVec4(int nameLocation) const;
    glm::mat2 getMat2(int nameLocation) const;
    glm::mat3 getMat3(int nameLocation) const;
    glm::mat4 getMat4(int nameLocation) const;
    int getTexture2D(int nameLocation, int texUnit) const;
};

template <> struct AssetType<Shader>
{
    static constexpr int type = PhysicsEngine::SHADER_TYPE;
};
template <> struct IsAssetInternal<Shader>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

namespace YAML
{
// RenderQueue
template <> struct convert<PhysicsEngine::RenderQueue>
{
    static Node encode(const PhysicsEngine::RenderQueue &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::RenderQueue &rhs)
    {
        rhs = static_cast<PhysicsEngine::RenderQueue>(node.as<int>());
        return true;
    }
};

// ShaderUniform
template <> struct convert<PhysicsEngine::ShaderUniform>
{
    static Node encode(const PhysicsEngine::ShaderUniform &rhs)
    {
        Node node;

        node["shortName"] = rhs.mShortName;
        node["blockName"] = rhs.mBlockName;
        node["nameLength"] = rhs.mNameLength;
        node["size"] = rhs.mSize;
        node["type"] = rhs.mType;
        node["variant"] = rhs.mVariant;
        node["location"] = rhs.mLocation;
        node["index"] = rhs.mIndex;

        /*if (rhs.mType == GL_INT)
        {
            node["data"] = *reinterpret_cast<const int*>(rhs.mData);
        }
        else if (rhs.mType == GL_FLOAT)
        {
            node["data"] = *reinterpret_cast<const float*>(rhs.mData);
        }
        else if (rhs.mType == GL_FLOAT_VEC2)
        {
            node["data"] = *reinterpret_cast<const glm::vec2*>(rhs.mData);
        }
        else if (rhs.mType == GL_FLOAT_VEC3)
        {
            node["data"] = *reinterpret_cast<const glm::vec3*>(rhs.mData);
        }
        else if (rhs.mType == GL_FLOAT_VEC4)
        {
            node["data"] = *reinterpret_cast<const glm::vec4*>(rhs.mData);
        }

        if (rhs.mType == GL_SAMPLER_2D)
        {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Guid*>(rhs.mData);
        }*/

        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShaderUniform &rhs)
    {
        std::string shortName = node["shortName"].as<std::string>();
        std::string blockName = node["blockName"].as<std::string>();

        memset(rhs.mShortName, '\0', 32);
        memset(rhs.mBlockName, '\0', 32);

        memcpy(rhs.mShortName, shortName.data(), std::min((size_t)32, shortName.length()));
        memcpy(rhs.mBlockName, blockName.data(), std::min((size_t)32, blockName.length()));

        rhs.mNameLength = node["nameLength"].as<size_t>();
        rhs.mSize = node["size"].as<size_t>();
        rhs.mType = node["type"].as<int>();
        rhs.mVariant = node["variant"].as<int>();
        rhs.mLocation = node["location"].as<int>();
        rhs.mIndex = node["index"].as<size_t>();

        /*if (rhs.mType == GL_INT)
        {
            int data = node["data"].as<int>();
            memcpy(rhs.mData, &data, sizeof(int));
        }
        else if (rhs.mType == GL_FLOAT)
        {
            float data = node["data"].as<float>();
            memcpy(rhs.mData, &data, sizeof(float));
        }
        else if (rhs.mType == GL_FLOAT_VEC2)
        {
            glm::vec2 data = node["data"].as<glm::vec2>();
            memcpy(rhs.mData, &data, sizeof(glm::vec2));
        }
        else if (rhs.mType == GL_FLOAT_VEC3)
        {
            glm::vec3 data = node["data"].as<glm::vec3>();
            memcpy(rhs.mData, &data, sizeof(glm::vec3));
        }
        else if (rhs.mType == GL_FLOAT_VEC4)
        {
            glm::vec4 data = node["data"].as<glm::vec4>();
            memcpy(rhs.mData, &data, sizeof(glm::vec4));
        }

        if (rhs.mType == GL_SAMPLER_2D)
        {
            PhysicsEngine::Guid data = node["data"].as<PhysicsEngine::Guid>();
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
        }*/

        return true;
    }
};
} // namespace YAML

#endif