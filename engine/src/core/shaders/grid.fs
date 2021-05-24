const std::string InternalShaders::gridFragmentShader =
"in vec4 Color;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	float depth = 0.2f * gl_FragCoord.z / gl_FragCoord.w;\n"
"	FragColor = vec4(Color.x, Color.y, Color.z, clamp(1.0f / depth, 0.0f, 0.8f));\n"
"}\n";
