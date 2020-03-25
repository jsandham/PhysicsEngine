#ifndef __FONT_H__
#define __FONT_H__

#include <map>
#include <vector>
#include <string>

#include "Asset.h"
#include "Shader.h"

#include "../graphics/GraphicsHandle.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct FontHeader
	{
		Guid mFontId;
		size_t mFilepathSize;
	};
#pragma pack(pop)

	struct Character
	{
		GraphicsHandle mGlyphId;    // ID handle of the glyph texture
	    glm::ivec2 mSize;     // Size of glyph
	    glm::ivec2 mBearing;  // Offset from baseline to left/top of glyph
	    unsigned int mAdvance;   
	};

	class Font : public Asset
	{
		private:
			std::string mFilepath;
			std::map<char, Character> mCharacters;

		public:
			Shader mShader;
			GraphicsHandle mVao;
			GraphicsHandle mVbo;

		public:
			Font();
			Font(std::string filepath);
			Font(std::vector<char> data);
			~Font();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			void load(std::string filepath);

			Character getCharacter(char c) const;
	};

	template <>
	const int AssetType<Font>::type = 6;

	template <typename T>
	struct IsFont { static const bool value; };

	template <typename T>
	const bool IsFont<T>::value = false;

	template<>
	const bool IsFont<Font>::value = true;
	template<>
	const bool IsAsset<Font>::value = true;
}

#endif