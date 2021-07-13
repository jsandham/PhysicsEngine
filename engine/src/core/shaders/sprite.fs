const std::string InternalShaders::spriteFragmentShader =
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D image;\n"
"uniform vec4 spriteColor;\n"
"void main()\n"
"{\n"
"    color = spriteColor * texture(image, TexCoords);\n"
"}\n";