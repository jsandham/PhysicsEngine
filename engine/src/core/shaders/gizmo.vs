const std::string InternalShaders::gizmoVertexShader =
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec3 normal;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    FragPos = vec3(model * vec4(position, 1.0));\n"
"    Normal = mat3(transpose(inverse(model))) * normal;\n"
"    gl_Position = projection * view * vec4(FragPos, 1.0);\n"
"}\n";

