const std::string InternalShaders::shadowDepthCubemapFragmentShader =
"in vec4 FragPos;\n"
"uniform vec3 lightPos;\n"
"uniform float farPlane;\n"
"void main()\n"
"{\n"
"	float lightDistance = length(FragPos.xyz - lightPos);\n"
"   lightDistance = lightDistance / farPlane;\n"
"   gl_FragDepth = 1.0f;\n"
"}\n";
