#include <iostream>

#include "../../include/core/Font.h"
#include "../../include/graphics/Graphics.h"

// #include "../../include/freetype/ft2build.h"
#include <ft2build.h>
#include FT_FREETYPE_H

using namespace PhysicsEngine;

Font::Font()
{
	assetId = Guid::INVALID;
}

Font::Font(std::string filepath)
{
	assetId = Guid::INVALID;
	this->filepath = filepath;
}

Font::Font(std::vector<char> data)
{
	int index = sizeof(int);
	FontHeader* header = reinterpret_cast<FontHeader*>(&data[index]);
	
	assetId = header->fontId;

	index += sizeof(FontHeader);

	if(header->filepathSize > 0 && header->filepathSize < 256){
		filepath = std::string(&data[index], header->filepathSize);

		std::cout << "font filepath loaded: " << filepath << std::endl;
	}
	else{
		std::cout << "Error: Font filepath size (" << header->filepathSize << ") is invalid" << std::endl;
		return; 
	}
}

Font::~Font()
{

}

void Font::load(std::string filepath)
{
    this->filepath = filepath;

	FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft)){
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
    	return;
    }

    std::cout << "loading font with filepath: " << filepath << std::endl;

    // Load font as face
    FT_Face face;
    if (FT_New_Face(ft, filepath.c_str(), 0, &face)){
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return;
    }

    // Set size to load glyphs as
    FT_Set_Pixel_Sizes(face, 0, 48);

    //TODO: Move into Graphics so Font does not depend on opengl
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
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );
        // Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Now store character for later use
        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            face->glyph->advance.x
        };

        //std::cout << "character: " << c << " " << face->glyph->bitmap.width << " " << face->glyph->bitmap.rows << std::endl;

        characters.insert(std::pair<GLchar, Character>(c, character));
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    shader.vertexShader = Shader::fontVertexShader;
    shader.fragmentShader = Shader::fontFragmentShader;

    shader.compile();

    glGenVertexArrays(1, &vao.handle);
    glGenBuffers(1, &vbo.handle);
    glBindVertexArray(vao.handle);
    glBindBuffer(GL_ARRAY_BUFFER, vbo.handle);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

Character Font::getCharacter(char c) const
{
	std::map<char, Character>::const_iterator it = characters.find(c);
	if(it != characters.end()){
		return it->second;
	}
	else{
		std::cout << "Error: Character " << c << " does not exist in font" << std::endl;
		Character character;
		return character;
	}
}