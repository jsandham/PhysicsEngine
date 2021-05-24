const std::string InternalShaders::ssaoVertexShader =
"in vec3 position;\n"
"in vec2 texCoord;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"   TexCoord = texCoord;\n"
"}\n";
