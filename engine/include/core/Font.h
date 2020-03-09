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
		Guid fontId;
		size_t filepathSize;
	};
#pragma pack(pop)

	struct Character
	{
		GraphicsHandle glyphId;    // ID handle of the glyph texture
	    glm::ivec2 size;     // Size of glyph
	    glm::ivec2 bearing;  // Offset from baseline to left/top of glyph
	    unsigned int advance;   
	};

	class Font : public Asset
	{
		private:
			std::string filepath;
			std::map<char, Character> characters;

		public:
			Shader shader;
			GraphicsHandle vao;
			GraphicsHandle vbo;

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
	struct IsFont { static bool value; };

	template <typename T>
	bool IsFont<T>::value = false;

	template<>
	bool IsFont<Font>::value = true;
	template<>
	bool IsAsset<Font>::value = true;
}

#endif