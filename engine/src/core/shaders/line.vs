const std::string InternalShaders::lineVertexShader =
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec4 color;\n"
"uniform mat4 mvp;\n"
"out vec4 Color;\n"
"void main()\n"
"{\n"
"   Color = color;\n"
"	gl_Position = mvp * vec4(position, 1.0);\n"
"}\n";
