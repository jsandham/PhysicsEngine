const std::string InternalShaders::normalMapFragmentShader =
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(Normal.xyz, 1.0f);\n"
"}\n";
