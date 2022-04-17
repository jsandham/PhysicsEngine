#ifndef SHADER_H__
#define SHADER_H__

#define NOMINMAX

#include <vector>
#include <unordered_map>
#include <set>
#include <cstdint>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "Asset.h"
#include "Color.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
enum class RenderQueue
{
    Opaque = 0,
    Transparent = 1
};

enum class ShaderUniformType
{
    Int = 0,
    Float = 1,
    Color = 2,
    Vec2 = 3,
    Vec3 = 4,
    Vec4 = 5,
    Mat2 = 6,
    Mat3 = 7,
    Mat4 = 8,
    Sampler2D = 9,
    SamplerCube = 10,
    Invalid = 11
};

enum class ShaderMacro
{
    None = 0,
    Directional = 1,
    Spot = 2,
    Point = 4,
    HardShadows = 8,
    SoftShadows = 16,
    SSAO = 32,
    ShowCascades = 64,
    Instancing = 128
};

enum class ShaderSourceLanguage
{
    GLSL = 0,
    HLSL = 1
};

struct ShaderCreationAttrib
{
    std::string mName;
    std::string mSourceFilepath;
    ShaderSourceLanguage mSourceLanguage;
    std::unordered_map<int, std::set<ShaderMacro>> mVariantMacroMap;
};

struct ShaderStatus
{
    char mVertexCompileLog[512];
    char mFragmentCompileLog[512];
    char mGeometryCompileLog[512];
    char mLinkLog[512];
    int mVertexShaderCompiled;
    int mFragmentShaderCompiled;
    int mGeometryShaderCompiled;
    int mShaderLinked;
};

struct ShaderProgram
{
    ShaderStatus mStatus;

    std::string mVertexShader;
    std::string mFragmentShader;
    std::string mGeometryShader;

    int64_t mVariant;
    unsigned int mHandle;
};

struct ShaderUniform
{
    char mData[64];
    std::string mName; // variable name in GLSL (including block name if applicable)
    ShaderUniformType mType; // type of the uniform (float, vec3 or mat4, etc)
    int mTex; // if data stores a texture id, this is the texture handle
    unsigned int mUniformId; // integer hash of uniform name

    std::string getShortName() const
    {
        size_t pos = mName.find_first_of('.');
        return mName.substr(pos + 1);
    }
};

struct ShaderAttribute
{
    std::string mName;
};

class Shader : public Asset
{
  private:
    std::string mSource;
    std::string mSourceFilepath;
      
    std::string mVertexShader;
    std::string mFragmentShader;
    std::string mGeometryShader;

    std::unordered_map<int, std::set<ShaderMacro>> mVariantMacroMap;

    std::vector<ShaderProgram> mPrograms;
    std::vector<ShaderUniform> mUniforms;
    std::vector<ShaderUniform> mMaterialUniforms;
    std::vector<ShaderAttribute> mAttributes;

    ShaderSourceLanguage mShaderSourceLanguage;

    bool mAllProgramsCompiled;
    int mActiveProgram;

  public:
    Shader(World *world);
    Shader(World *world, Id id);
    ~Shader();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const ShaderCreationAttrib& attrib);

    bool isCompiled() const;

    void addVariant(int variantId, const std::set<ShaderMacro>& macros);
    void preprocess();
    void compile();
    void use(int program);
    void unuse();
    void setVertexShader(const std::string &vertexShader);
    void setGeometryShader(const std::string &geometryShader);
    void setFragmentShader(const std::string &fragmentShader);
    void setUniformBlock(const std::string &blockName, int bindingPoint) const;
    int findUniformLocation(const std::string &name, int program) const;
    int getProgramFromVariant(int64_t variant) const;
    int getActiveProgram() const;

    std::vector<ShaderProgram> getPrograms() const;
    std::vector<ShaderUniform> getUniforms() const;
    std::vector<ShaderUniform> getMaterialUniforms() const;
    std::vector<ShaderAttribute> getAttributeNames() const;
    std::string getVertexShader() const;
    std::string getGeometryShader() const;
    std::string getFragmentShader() const;
    std::string getSource() const;
    std::string getSourceFilepath() const;
    ShaderSourceLanguage getSourceLanguage() const;

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
    void setTexture2Ds(const char *name, int *texUnits, int count, int* texs) const;

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
    void setTexture2Ds(int nameLocation, int *texUnits, int count, int *texs) const;

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

    static unsigned int uniformToId(const char* property);
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
        switch (rhs)
        {
        case PhysicsEngine::RenderQueue::Opaque:
            node = "Opaque";
            break;
        case PhysicsEngine::RenderQueue::Transparent:
            node = "Transparent";
            break;
        }

        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::RenderQueue &rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Opaque")
        {
            rhs = PhysicsEngine::RenderQueue::Opaque;
        }
        else if (type == "Transparent")
        {
            rhs = PhysicsEngine::RenderQueue::Transparent;
        }

        return true;
    }
};

// ShaderSourceLanguage
template <> struct convert<PhysicsEngine::ShaderSourceLanguage>
{
    static Node encode(const PhysicsEngine::ShaderSourceLanguage& rhs)
    {
        Node node;
        switch (rhs)
        {
        case PhysicsEngine::ShaderSourceLanguage::GLSL:
            node = "GLSL";
            break;
        case PhysicsEngine::ShaderSourceLanguage::HLSL:
            node = "HLSL";
            break;
        }

        return node;
    }

    static bool decode(const Node& node, PhysicsEngine::ShaderSourceLanguage& rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "GLSL")
        {
            rhs = PhysicsEngine::ShaderSourceLanguage::GLSL;
        }
        else if (type == "HLSL")
        {
            rhs = PhysicsEngine::ShaderSourceLanguage::HLSL;
        }

        return true;
    }
};

// ShaderUniformType
template <> struct convert<PhysicsEngine::ShaderUniformType>
{
    static Node encode(const PhysicsEngine::ShaderUniformType& rhs)
    {
        Node node;

        switch (rhs)
        {
        case PhysicsEngine::ShaderUniformType::Int:
            node = "Int";
            break;
        case PhysicsEngine::ShaderUniformType::Float:
            node = "Float";
            break;
        case PhysicsEngine::ShaderUniformType::Color:
            node = "Color";
            break;
        case PhysicsEngine::ShaderUniformType::Vec2:
            node = "Vec2";
            break;
        case PhysicsEngine::ShaderUniformType::Vec3:
            node = "Vec3";
            break;
        case PhysicsEngine::ShaderUniformType::Vec4:
            node = "Vec4";
            break;
        case PhysicsEngine::ShaderUniformType::Mat2:
            node = "Mat2";
            break;
        case PhysicsEngine::ShaderUniformType::Mat3:
            node = "Mat3";
            break;
        case PhysicsEngine::ShaderUniformType::Mat4:
            node = "Mat4";
            break;
        case PhysicsEngine::ShaderUniformType::Sampler2D:
            node = "Sampler2D";
            break;
        case PhysicsEngine::ShaderUniformType::SamplerCube:
            node = "SamplerCube";
            break;
        default:
            node = "Invalid";
            break;
        }
      
        return node;
    }

    static bool decode(const Node& node, PhysicsEngine::ShaderUniformType& rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Int") {
            rhs = PhysicsEngine::ShaderUniformType::Int;
        }
        else if (type == "Float"){
            rhs = PhysicsEngine::ShaderUniformType::Float;
        }
        else if (type == "Color")
        {
            rhs = PhysicsEngine::ShaderUniformType::Color;
        }
        else if (type == "Vec2"){
            rhs = PhysicsEngine::ShaderUniformType::Vec2;
        }
        else if (type == "Vec3"){
            rhs = PhysicsEngine::ShaderUniformType::Vec3;
        }
        else if (type == "Vec4"){
            rhs = PhysicsEngine::ShaderUniformType::Vec4;
        }
        else if (type == "Mat2"){
            rhs = PhysicsEngine::ShaderUniformType::Mat2;
        }
        else if (type == "Mat3"){
            rhs = PhysicsEngine::ShaderUniformType::Mat3;
        }
        else if (type == "Mat4"){
            rhs = PhysicsEngine::ShaderUniformType::Mat4;
        }
        else if (type == "Sampler2D"){
            rhs = PhysicsEngine::ShaderUniformType::Sampler2D;
        }
        else if (type == "SamplerCube"){
            rhs = PhysicsEngine::ShaderUniformType::SamplerCube;
        }
        else {
            rhs = PhysicsEngine::ShaderUniformType::Invalid;
        }
        
        return true;
    }
};

// ShaderMacro
template <> struct convert<PhysicsEngine::ShaderMacro>
{
    static Node encode(const PhysicsEngine::ShaderMacro& rhs)
    {
        Node node;

        switch (rhs)
        {
        case PhysicsEngine::ShaderMacro::Directional:
            node = "Directional";
            break;
        case PhysicsEngine::ShaderMacro::Spot:
            node = "Spot";
            break;
        case PhysicsEngine::ShaderMacro::Point:
            node = "Point";
            break;
        case PhysicsEngine::ShaderMacro::HardShadows:
            node = "HardShadows";
            break;
        case PhysicsEngine::ShaderMacro::SoftShadows:
            node = "SoftShadows";
            break;
        case PhysicsEngine::ShaderMacro::SSAO:
            node = "SSAO";
            break;
        case PhysicsEngine::ShaderMacro::ShowCascades:
            node = "ShowCascades";
            break;
        case PhysicsEngine::ShaderMacro::Instancing:
            node = "Instancing";
            break;
        default:
            node = "None";
            break;
        }

        return node;
    }

    static bool decode(const Node& node, PhysicsEngine::ShaderMacro& rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Directional") {
            rhs = PhysicsEngine::ShaderMacro::Directional;
        }
        else if (type == "Spot") {
            rhs = PhysicsEngine::ShaderMacro::Spot;
        }
        else if (type == "Point") {
            rhs = PhysicsEngine::ShaderMacro::Point;
        }
        else if (type == "HardShadows") {
            rhs = PhysicsEngine::ShaderMacro::HardShadows;
        }
        else if (type == "SoftShadows") {
            rhs = PhysicsEngine::ShaderMacro::SoftShadows;
        }
        else if (type == "SSAO") {
            rhs = PhysicsEngine::ShaderMacro::SSAO;
        }
        else if (type == "ShowCascades") {
            rhs = PhysicsEngine::ShaderMacro::ShowCascades;
        }
        else if (type == "Instancing")
        {
            rhs = PhysicsEngine::ShaderMacro::Instancing;
        }
        else {
            rhs = PhysicsEngine::ShaderMacro::None;
        }

        return true;
    }
};

// ShaderUniform
template <> struct convert<PhysicsEngine::ShaderUniform>
{
    static Node encode(const PhysicsEngine::ShaderUniform &rhs)
    {
        Node node;

        node["type"] = rhs.mType;

        if (rhs.mType == PhysicsEngine::ShaderUniformType::Int)
        {
            node["data"] = *reinterpret_cast<const int *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Float)
        {
            node["data"] = *reinterpret_cast<const float *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Color)
        {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Color *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec2)
        {
            node["data"] = *reinterpret_cast<const glm::vec2 *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec3)
        {
            node["data"] = *reinterpret_cast<const glm::vec3 *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec4)
        {
            node["data"] = *reinterpret_cast<const glm::vec4 *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Sampler2D)
        {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Guid *>(rhs.mData);
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::SamplerCube)
        {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Guid*>(rhs.mData);
        }

        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShaderUniform &rhs)
    {
        rhs.mType = YAML::getValue<PhysicsEngine::ShaderUniformType>(node, "type");

        if (rhs.mType == PhysicsEngine::ShaderUniformType::Int)
        {
            int data = YAML::getValue<int>(node, "data");
            memcpy(rhs.mData, &data, sizeof(int));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Float)
        {
            float data = YAML::getValue<float>(node, "data");
            memcpy(rhs.mData, &data, sizeof(float));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Color)
        {
            PhysicsEngine::Color data = YAML::getValue<PhysicsEngine::Color>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Color));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec2)
        {
            glm::vec2 data = YAML::getValue<glm::vec2>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec2));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec3)
        {
            glm::vec3 data = YAML::getValue<glm::vec3>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec3));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Vec4)
        {
            glm::vec4 data = YAML::getValue<glm::vec4>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec4));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::Sampler2D)
        {
            PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
        }
        else if (rhs.mType == PhysicsEngine::ShaderUniformType::SamplerCube)
        {
            PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
        }

        return true;
    }
};

// std::set<ShaderMacro>
template <> struct convert<std::set<PhysicsEngine::ShaderMacro>>
{
    static Node encode(const std::set<PhysicsEngine::ShaderMacro>& rhs)
    {
        Node node = YAML::Load("[]");

        for (auto it = rhs.begin(); it != rhs.end(); it++)
        {
            node.push_back(*it);
        }

        return node;
    }

    static bool decode(const Node& node, std::set<PhysicsEngine::ShaderMacro>& rhs)
    {
        if (!node.IsSequence())
        {
            return false;
        }

        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            rhs.insert(it->as<PhysicsEngine::ShaderMacro>());
        }

        return true;
    }
};

// std::unordered_map<int, std::set<ShaderMacro>>
template <> struct convert<std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>>>
{
    static Node encode(const std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>>& rhs)
    {
        Node node;

        for (auto it = rhs.begin(); it != rhs.end(); it++)
        {
            node[std::to_string(it->first)] = it->second;
        }

        return node;
    }

    static bool decode(const Node& node, std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>>& rhs)
    {
        if (!node.IsMap())
        {
            return false;
        }

        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            rhs[it->first.as<int>()] = it->second.as<std::set<PhysicsEngine::ShaderMacro>>();
        }

        return true;
    }
};
} // namespace YAML

#endif