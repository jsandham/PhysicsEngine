const std::string InternalShaders::gizmoFragmentShader =
"out vec4 FragColor;\n"
"in vec3 Normal;\n"
"in vec3 FragPos;\n"
"uniform vec3 lightPos;\n"
"uniform vec4 color;\n"
"void main()\n"
"{\n"
"    vec3 norm = normalize(Normal);\n"
"    vec3 lightDir = normalize(lightPos - FragPos);\n"
"    float diff = max(abs(dot(norm, lightDir)), 0.1);\n"
"    vec4 diffuse = vec4(diff, diff, diff, 1.0);\n"
"    FragColor = diffuse * color;\n"
"}\n";
