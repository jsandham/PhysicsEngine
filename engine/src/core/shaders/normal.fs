const std::string InternalShaders::normalFragmentShader =
"uniform int wireframe;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n"
"void main(void)\n"
"{\n"
"	FragColor = vec4(wireframe * Normal, 1.0f);\n"
"}\n";
