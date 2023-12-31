#ifndef SHADER_H__
#define SHADER_H__

#define NOMINMAX

#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "glm.h"
#include "SerializationEnums.h"
#include "Color.h"
#include "AssetEnums.h"
#include "Guid.h"
#include "Id.h"

#include "../graphics/ShaderProgram.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
struct ShaderCreationAttrib
{
    std::string mName;
    std::string mSourceFilepath;
    std::unordered_map<int, std::set<ShaderMacro>> mVariantMacroMap;
};

class World;

class Shader
{
  private:
    Guid mGuid;
    Id mId;
    World *mWorld;

    std::string mSource;
    std::string mSourceFilepath;

    std::string mVertexShader;
    std::string mFragmentShader;
    std::string mGeometryShader;

    std::unordered_map<int, std::set<ShaderMacro>> mVariantMacroMap;

    std::vector<ShaderProgram *> mPrograms;
    std::vector<int64_t> mVariants;

    std::vector<ShaderUniform> mUniforms;
    std::vector<ShaderUniform> mMaterialUniforms;

    bool mAllProgramsCompiled;
    ShaderProgram *mActiveProgram;

    friend class World;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    Shader(World *world, const Id &id);
    Shader(World *world, const Guid &guid, const Id &id);
    ~Shader();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    bool writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void load(const ShaderCreationAttrib &attrib);

    bool isCompiled() const;

    void addVariant(int variantId, const std::set<ShaderMacro> &macros);
    void removeVariant(int variantId);
    void preprocess();
    void compile();
    void bind(int64_t variant);
    void unbind();
    void setVertexShader(const std::string &vertexShader);
    void setGeometryShader(const std::string &geometryShader);
    void setFragmentShader(const std::string &fragmentShader);

    ShaderProgram *getProgramFromVariant(int64_t variant) const;
    ShaderProgram *getActiveProgram() const;

    std::vector<ShaderProgram *> getPrograms() const;
    std::vector<ShaderUniform> getUniforms() const;
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
    void setTexture2D(const char *name, int texUnit, void *tex) const;
    void setTexture2Ds(const char *name, const std::vector<int> &texUnits, int count,
                       const std::vector<void *> &texs) const;

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
    void setTexture2D(int uniformId, int texUnit, void *tex) const;
    void setTexture2Ds(int uniformId, const std::vector<int> &texUnits, int count,
                       const std::vector<void *> &texs) const;

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

    static int uniformToId(const char *property);

    static int MODEL_UNIFORM_ID;
    static int SHADOW_MAP_UNIFORM_ID;
};

} // namespace PhysicsEngine

#endif