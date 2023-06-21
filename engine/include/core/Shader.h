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

#include "../graphics/ShaderProgram.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
enum class RenderQueue
{
    Opaque = 0,
    Transparent = 1
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

struct ShaderCreationAttrib
{
    std::string mName;
    std::string mSourceFilepath;
    std::unordered_map<int, std::set<ShaderMacro>> mVariantMacroMap;
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

    std::vector<ShaderProgram*> mPrograms;
    std::vector<int64_t> mVariants;

    std::vector<ShaderUniform> mMaterialUniforms;

    bool mAllProgramsCompiled;
    ShaderProgram *mActiveProgram;

  public:
    Shader(World *world, const Id &id);
    Shader(World *world, const Guid &guid, const Id &id);
    ~Shader();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const ShaderCreationAttrib& attrib);

    bool isCompiled() const;

    void addVariant(int variantId, const std::set<ShaderMacro>& macros);
    void removeVariant(int variantId);
    void preprocess();
    void compile();
    void bind(int64_t variant);
    void unbind();
    void setVertexShader(const std::string &vertexShader);
    void setGeometryShader(const std::string &geometryShader);
    void setFragmentShader(const std::string &fragmentShader);
    
    ShaderProgram* getProgramFromVariant(int64_t variant) const;
    ShaderProgram* getActiveProgram() const;

    std::vector<ShaderProgram*> getPrograms() const;
    std::vector<ShaderUniform> getMaterialUniforms() const;
    std::string getVertexShader() const;
    std::string getGeometryShader() const;
    std::string getFragmentShader() const;
    std::string getSource() const;
    std::string getSourceFilepath() const;

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
    void setTexture2D(const char *name, int texUnit, void* tex) const;
    void setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<void*>& texs) const;

    void setBool(int uniformId, bool value) const;
    void setInt(int uniformId, int value) const;
    void setFloat(int uniformId, float value) const;
    void setColor(int uniformId, const Color &color) const;
    void setVec2(int uniformId, const glm::vec2 &vec) const;
    void setVec3(int uniformId, const glm::vec3 &vec) const;
    void setVec4(int uniformId, const glm::vec4 &vec) const;
    void setMat2(int uniformId, const glm::mat2 &mat) const;
    void setMat3(int uniformId, const glm::mat3 &mat) const;
    void setMat4(int uniformId, const glm::mat4 &mat) const;
    void setTexture2D(int uniformId, int texUnit, void* tex) const;
    void setTexture2Ds(int uniformId, const std::vector<int>& texUnits, int count, const std::vector<void*>& texs) const;

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

    bool getBool(int uniformId) const;
    int getInt(int uniformId) const;
    float getFloat(int uniformId) const;
    Color getColor(int uniformId) const;
    glm::vec2 getVec2(int uniformId) const;
    glm::vec3 getVec3(int uniformId) const;
    glm::vec4 getVec4(int uniformId) const;
    glm::mat2 getMat2(int uniformId) const;
    glm::mat3 getMat3(int uniformId) const;
    glm::mat4 getMat4(int uniformId) const;

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

        switch (rhs.mType)
        {
        case PhysicsEngine::ShaderUniformType::Int: {
            node["data"] = *reinterpret_cast<const int *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Float: {
            node["data"] = *reinterpret_cast<const float *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Color: {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Color *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec2: {
            node["data"] = *reinterpret_cast<const glm::vec2 *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec3: {
            node["data"] = *reinterpret_cast<const glm::vec3 *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec4: {
            node["data"] = *reinterpret_cast<const glm::vec4 *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::Sampler2D: {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Guid *>(rhs.mData);
            break;
        }
        case PhysicsEngine::ShaderUniformType::SamplerCube: {
            node["data"] = *reinterpret_cast<const PhysicsEngine::Guid *>(rhs.mData);
            break;
        }
        }

        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShaderUniform &rhs)
    {
        rhs.mType = YAML::getValue<PhysicsEngine::ShaderUniformType>(node, "type");

        switch (rhs.mType)
        {
        case PhysicsEngine::ShaderUniformType::Int: {
            int data = YAML::getValue<int>(node, "data");
            memcpy(rhs.mData, &data, sizeof(int));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Float: {
            float data = YAML::getValue<float>(node, "data");
            memcpy(rhs.mData, &data, sizeof(float));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Color: {
            PhysicsEngine::Color data = YAML::getValue<PhysicsEngine::Color>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Color));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec2: {
            glm::vec2 data = YAML::getValue<glm::vec2>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec2));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec3: {
            glm::vec3 data = YAML::getValue<glm::vec3>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec3));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Vec4: {
            glm::vec4 data = YAML::getValue<glm::vec4>(node, "data");
            memcpy(rhs.mData, &data, sizeof(glm::vec4));
            break;
        }
        case PhysicsEngine::ShaderUniformType::Sampler2D: {
            PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
            break;
        }
        case PhysicsEngine::ShaderUniformType::SamplerCube: {
            PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
            memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
            break;
        }
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