const std::string InternalShaders::shadowDepthMapVertexShader =
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = projection * view * model * vec4(position, 1.0);\n"
"}\n";
