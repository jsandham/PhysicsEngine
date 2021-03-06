#include <iostream>
#include <algorithm>

#include "../../include/core/Font.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/core/Serialize.h"
#include "../../include/graphics/Graphics.h"

// #include "../../include/freetype/ft2build.h"
#include <ft2build.h>
#include FT_FREETYPE_H

using namespace PhysicsEngine;

Font::Font() : Asset()
{
}

Font::Font(Guid id) : Asset(id)
{
}

Font::Font(const std::string &filepath) : Asset()
{
    mFilepath = filepath;
}

Font::~Font()
{
}

std::vector<char> Font::serialize() const
{
    return serialize(mId);
}

std::vector<char> Font::serialize(Guid assetId) const
{
    FontHeader header;
    header.mFontId = assetId;
    header.mFilepathSize = mFilepath.length();

    std::size_t len = std::min(size_t(64 - 1), mAssetName.size());
    memcpy(&header.mFontName[0], &mAssetName[0], len);
    header.mFontName[len] = '\0';

    size_t numberOfBytes = sizeof(FontHeader) + sizeof(char) * mFilepath.length();

    std::vector<char> data(numberOfBytes);

    size_t start1 = 0;
    size_t start2 = start1 + sizeof(FontHeader);

    memcpy(&data[start1], &header, sizeof(FontHeader));
    memcpy(&data[start2], mFilepath.c_str(), sizeof(char) * mFilepath.length());

    return data;
}

void Font::deserialize(const std::vector<char> &data)
{
    int index = sizeof(int);
    const FontHeader *header = reinterpret_cast<const FontHeader *>(&data[index]);

    mId = header->mFontId;
    mAssetName = std::string(header->mFontName);

    index += sizeof(FontHeader);

    if (header->mFilepathSize > 0 && header->mFilepathSize < 256)
    {
        mFilepath = std::string(&data[index], header->mFilepathSize);

        std::cout << "font filepath loaded: " << mFilepath << std::endl;
    }
    else
    {
        std::cout << "Error: Font filepath size (" << header->mFilepathSize << ") is invalid" << std::endl;
        return;
    }
}

void Font::serialize(std::ostream& out) const
{
    Asset::serialize(out);
}

void Font::deserialize(std::istream& in)
{
    Asset::deserialize(in);
}

void Font::load(std::string filepath)
{
    mFilepath = filepath;

    FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft))
    {
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
        return;
    }

    std::cout << "loading font with filepath: " << filepath << std::endl;

    // Load font as face
    FT_Face face;
    if (FT_New_Face(ft, filepath.c_str(), 0, &face))
    {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return;
    }

    // Set size to load glyphs as
    FT_Set_Pixel_Sizes(face, 0, 48);

    // TODO: Move into Graphics so Font does not depend on opengl
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Load first 128 characters of ASCII set
    for (unsigned char c = 0; c < 128; c++)
    {
        // Load character glyph
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))
        {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            continue;
        }

        // TODO: Figure out how to move this out of Font class so that Font does not depend on OpenGL
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows, 0, GL_RED,
                     GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);
        // Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Now store character for later use
        Character character = {texture, glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                               glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                               (unsigned int)face->glyph->advance.x};

        mCharacters.insert(std::pair<GLchar, Character>(c, character));
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    mShader.setVertexShader(InternalShaders::fontVertexShader);
    mShader.setFragmentShader(InternalShaders::fontFragmentShader);

    mShader.compile();

    glGenVertexArrays(1, &mVao);
    glGenBuffers(1, &mVbo);
    glBindVertexArray(mVao);
    glBindBuffer(GL_ARRAY_BUFFER, mVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

Character Font::getCharacter(char c) const
{
    std::map<char, Character>::const_iterator it = mCharacters.find(c);
    if (it != mCharacters.end())
    {
        return it->second;
    }
    else
    {
        std::cout << "Error: Character " << c << " does not exist in font" << std::endl;
        Character character;
        return character;
    }
}