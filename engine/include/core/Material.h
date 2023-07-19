#ifndef MATERIAL_H__
#define MATERIAL_H__

#include "Asset.h"
#include "Shader.h"
#include "Texture2D.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
class Material : public Asset
{
  private:
    Guid mShaderGuid;
    std::vector<ShaderUniform> mUniforms;
    bool mShaderChanged;
    bool mTextureChanged;

    Shader *mShader;

  public:
    RenderQueue mRenderQueue;
    bool mEnableInstancing;

  public:
    Material(World *world, const Id &id);
    Material(World *world, const Guid &guid, const Id &id);
    ~Material();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void apply();
    void onShaderChanged();
    void onTextureChanged();
    bool hasShaderChanged() const;
    bool hasTextureChanged() const;
    void setShaderId(const Guid &shaderId);
    Guid getShaderId() const;
    std::vector<ShaderUniform> getUniforms() const;

    void setBool(const std::string &name, bool value);
    void setInt(const std::string &name, int value);
    void setFloat(const std::string &name, float value);
    void setColor(const std::string &name, const Color &color);
    void setVec2(const std::string &name, const glm::vec2 &vec);
    void setVec3(const std::string &name, const glm::vec3 &vec);
    void setVec4(const std::string &name, const glm::vec4 &vec);
    void setMat2(const std::string &name, const glm::mat2 &mat);
    void setMat3(const std::string &name, const glm::mat3 &mat);
    void setMat4(const std::string &name, const glm::mat4 &mat);
    void setTexture(const std::string &name, const Guid &textureId);

    void setBool(int uniformId, bool value);
    void setInt(int uniformId, int value);
    void setFloat(int uniformId, float value);
    void setColor(int uniformId, const Color &color);
    void setVec2(int uniformId, const glm::vec2 &vec);
    void setVec3(int uniformId, const glm::vec3 &vec);
    void setVec4(int uniformId, const glm::vec4 &vec);
    void setMat2(int uniformId, const glm::mat2 &mat);
    void setMat3(int uniformId, const glm::mat3 &mat);
    void setMat4(int uniformId, const glm::mat4 &mat);
    void setTexture(int uniformId, const Guid &textureId);

    bool getBool(const std::string &name) const;
    int getInt(const std::string &name) const;
    float getFloat(const std::string &name) const;
    Color getColor(const std::string &name) const;
    glm::vec2 getVec2(const std::string &name) const;
    glm::vec3 getVec3(const std::string &name) const;
    glm::vec4 getVec4(const std::string &name) const;
    glm::mat2 getMat2(const std::string &name) const;
    glm::mat3 getMat3(const std::string &name) const;
    glm::mat4 getMat4(const std::string &name) const;
    Guid getTexture(const std::string &name) const;

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
    Guid getTexture(int uniformId) const;

    std::vector<Guid> getTextures() const;

  private:
    int findIndexOfUniform(const std::string &name) const;
    int findIndexOfUniform(int uniformId) const;
};

template <> struct AssetType<Material>
{
    static constexpr int type = PhysicsEngine::MATERIAL_TYPE;
};

template <> struct IsAssetInternal<Material>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif