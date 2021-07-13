const std::string InternalShaders::spriteVertexShader =
"layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>\n"
"out vec2 TexCoords;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    TexCoords = vertex.zw;\n"
"    gl_Position = projection * view * model * vec4(vertex.xy, 0.0, 1.0);\n"
"}\n";