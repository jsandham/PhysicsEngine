const std::string InternalShaders::gridVertexShader =
"layout(std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 mvp;\n"
"uniform vec4 color;\n"
"in vec3 position;\n"
"out vec4 Color;\n"
"void main()\n"
"{\n"
"	gl_Position = mvp * vec4(position, 1.0);\n"
"	Color = color;\n"
"}\n";
