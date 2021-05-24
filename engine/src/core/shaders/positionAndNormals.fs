const std::string InternalShaders::positionAndNormalsFragmentShader =
"layout (location = 0) out vec3 positionTex;\n"
"layout (location = 1) out vec3 normalTex;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"void main()\n"
"{\n"
"    // store the fragment position vector in the first gbuffer texture\n"
"    positionTex = FragPos.xyz;\n"
"    // also store the per-fragment normals into the gbuffer\n"
"    normalTex = normalize(Normal);\n"
"}\n";
