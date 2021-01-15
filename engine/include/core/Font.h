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
#pragma pack(push, 1)
struct FontHeader
{
    Guid mFontId;
    char mFontName[64];
    size_t mFilepathSize;
};
#pragma pack(pop)

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

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid assetId) const;
    void deserialize(const std::vector<char> &data);

    void load(std::string filepath);

    Character getCharacter(char c) const;
};

template <typename T> struct IsFont
{
    static constexpr bool value = false;
};

template <> struct AssetType<Font>
{
    static constexpr int type = PhysicsEngine::FONT_TYPE;
};
template <> struct IsFont<Font>
{
    static constexpr bool value = true;
};
template <> struct IsAsset<Font>
{
    static constexpr bool value = true;
};
template <> struct IsAssetInternal<Font>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif