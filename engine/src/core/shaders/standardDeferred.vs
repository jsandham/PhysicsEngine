const std::string InternalShaders::standardDeferredVertexShader =
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec2 aTexCoords;\n"
"out vec2 TexCoords;\n"
"void main()\n"
"{\n"
"	TexCoords = aTexCoords;\n"
"	gl_Position = vec4(aPos, 1.0);\n"
"}\n";