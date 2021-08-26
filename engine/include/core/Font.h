#ifndef FONT_H__
#define FONT_H__

#include <map>
#include <string>
#include <vector>

#include "Asset.h"
#include "Shader.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"

namespace PhysicsEngine
{
struct Character
{
    GLuint mGlyphId;     // ID handle of the glyph texture
    glm::ivec2 mSize;    // Size of glyph
    glm::ivec2 mBearing; // Offset from baseline to left/top of glyph
    unsigned int mAdvance;
};

class Font : public Asset
{
  private:
    std::string mFilepath;
    std::map<char, Character> mCharacters;

  public:
    // Shader mShader;
    GLuint mVao;
    GLuint mVbo;

  public:
    Font(World *world);
    Font(World *world, Guid id);
    Font(World *world, const std::string &filepath);
    ~Font();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(std::string filepath);

    Character getCharacter(char c) const;
};

template <> struct AssetType<Font>
{
    static constexpr int type = PhysicsEngine::FONT_TYPE;
};
template <> struct IsAssetInternal<Font>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif