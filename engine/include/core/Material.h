#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Asset.h"
#include "Guid.h"
#include "Shader.h"
#include "Texture2D.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Material : public Asset
{
  private:
    Guid mShaderId;
    bool mShaderChanged;
    std::vector<ShaderUniform>
        mUniforms; // current storing all uniforms but maybe we should just hold the material uniforms herre?

  public:
    RenderQueue mRenderQueue;

  public:
    Material();
    Material(Guid id);
    ~Material();

    virtual void serialize(std::ostream& out) const;
    virtual void deserialize(std::istream& in);

    void load(const std::string &filepath);
    void load(Guid shaderId);
    void apply(World *world);
    void onShaderChanged(World *world);
    bool hasShaderChanged() const;

    void setShaderId(Guid shaderId);
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

    void setBool(int nameLocation, bool value);
    void setInt(int nameLocation, int value);
    void setFloat(int nameLocation, float value);
    void setColor(int nameLocation, const Color &color);
    void setVec2(int nameLocation, const glm::vec2 &vec);
    void setVec3(int nameLocation, const glm::vec3 &vec);
    void setVec4(int nameLocation, const glm::vec4 &vec);
    void setMat2(int nameLocation, const glm::mat2 &mat);
    void setMat3(int nameLocation, const glm::mat3 &mat);
    void setMat4(int nameLocation, const glm::mat4 &mat);
    void setTexture(int nameLocation, const Guid &textureId);

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
    Guid getTexture(int nameLocation) const;

    std::vector<Guid> getTextures() const;

  private:
    int findIndexOfUniform(const std::string &name) const;
    int findIndexOfUniform(int nameLocation) const;
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