#ifndef __FONT_H__
#define __FONT_H__

#include <map>
#include <string>
#include <vector>

#include "Asset.h"
#include "Shader.h"

#include "../glm/glm.hpp"

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
    Shader mShader;
    GLuint mVao;
    GLuint mVbo;

  public:
    Font();
    Font(Guid id);
    Font(const std::string &filepath);
    ~Font();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;

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