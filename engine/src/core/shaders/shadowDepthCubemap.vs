const std::string InternalShaders::shadowDepthCubemapVertexShader =
"in vec3 position;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"	gl_Position = model * vec4(position, 1.0);\n"
"}\n";
