#ifndef __FONT_H__
#define __FONT_H__

#include <map>
#include <vector>
#include <string>

#include "Asset.h"

#include "../graphics/GLHandle.h"

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
		GLHandle glyphId;    // ID handle of the glyph texture
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
			Font();
			Font(std::string filepath);
			Font(std::vector<char> data);
			~Font();

			void load();

			Character getCharacter(char c) const;
	};
}

#endif