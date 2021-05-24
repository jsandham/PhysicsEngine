const std::string InternalShaders::screenQuadFragmentShader =
"uniform sampler2D texture0;\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = texture(texture0, TexCoord);\n"
"}\n";
