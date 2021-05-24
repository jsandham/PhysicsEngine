const std::string InternalShaders::normalVertexShader =
"layout(std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"in vec2 texCoord;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"	FragPos = vec3(model * vec4(position, 1.0));\n"
"	Normal = normalize(normal);\n"
"	TexCoord = texCoord;\n"
"	gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);\n"
"}\n";
