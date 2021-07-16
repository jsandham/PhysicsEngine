const std::string InternalShaders::standardFragmentShader =
"layout(std140) uniform LightBlock\n"
"{\n"
"	mat4 lightProjection[5]; // 0    64   128  192  256\n"
"	mat4 lightView[5];       // 320  384  448  512  576\n"
"	vec3 position;           // 640\n"
"	vec3 direction;          // 656\n"
"	vec3 color;              // 672\n"
"	float cascadeEnds[5];    // 688  704  720  736  752\n"
"	float intensity;         // 768\n"
"	float spotAngle;         // 772\n"
"	float innerSpotAngle;    // 776\n"
"	float shadowNearPlane;   // 780\n"
"	float shadowFarPlane;    // 784\n"
"	float shadowAngle;       // 788\n"
"	float shadowRadius;      // 792\n"
"	float shadowStrength;    // 796\n"
"}Light;\n"
"struct Material\n"
"{\n"
"	float shininess;\n"
"	vec3 ambient;\n"
"	vec3 diffuse;\n"
"	vec3 specular;\n"
"	sampler2D mainTexture;\n"
"	sampler2D normalMap;\n"
"	sampler2D specularMap;\n"
"};\n"
"uniform Material material;\n"
"uniform sampler2D shadowMap[5];\n"
"in vec3 FragPos;\n"
"in vec3 CameraPos;\n"
"in vec3 Normal;\n"
"in vec2 TexCoord;\n"
"in float ClipSpaceZ;\n"
"in vec4 FragPosLightSpace[5];\n"
"vec2 poissonDisk[4] = vec2[](\n"
"	vec2(-0.94201624, -0.39906216),\n"
"	vec2(0.94558609, -0.76890725),\n"
"	vec2(-0.094184101, -0.92938870),\n"
"	vec2(0.34495938, 0.29387760)\n"
"	);\n"
"out vec4 FragColor;\n"
"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);\n"
"vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);\n"
"vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);\n"
"float CalcShadow(int index, vec4 fragPosLightSpace);\n"
"void main(void)\n"
"{\n"
"	vec3 viewDir = normalize(CameraPos - FragPos);\n"
"//#if defined(DIRECTIONALLIGHT)\n"
"	FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
"//#elif defined(SPOTLIGHT)\n"
"//	FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
"//#elif defined(POINTLIGHT)\n"
"//	FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
"//#else\n"
"//	FragColor = vec4(0.5, 0.5, 0.5, 1.0);\n"
"//#endif\n"
"//#if defined(DIRECTIONALLIGHT) && defined(CASCADE)\n"
"//	if (ClipSpaceZ <= Light.cascadeEnds[0]) {\n"
"//		FragColor = FragColor * vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"//	}\n"
"//	else if (ClipSpaceZ <= Light.cascadeEnds[1]) {\n"
"//		FragColor = FragColor * vec4(0.0f, 1.0f, 0.0f, 1.0f);\n"
"//	}\n"
"//	else if (ClipSpaceZ <= Light.cascadeEnds[2]) {\n"
"//		FragColor = FragColor * vec4(0.0f, 0.0f, 1.0f, 1.0f);\n"
"//	}\n"
"//	else if (ClipSpaceZ <= Light.cascadeEnds[3]) {\n"
"//		FragColor = FragColor * vec4(0.0f, 1.0f, 1.0f, 1.0f);\n"
"//	}\n"
"//	else if (ClipSpaceZ <= Light.cascadeEnds[4]) {\n"
"//		FragColor = FragColor * vec4(0.6f, 0.0f, 0.6f, 1.0f);\n"
"//	}\n"
"//	else {\n"
"//		FragColor = vec4(0.5, 0.5, 0.5, 1.0);\n"
"//	}\n"
"//#endif\n"
"}\n"
"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)\n"
"{\n"
"	vec3 norm = normalize(normal);\n"
"	vec3 lightDir = normalize(Light.direction);\n"
"	vec3 reflectDir = reflect(-lightDir, norm);\n"
"	float ambientStrength = 1.0f;\n"
"	float diffuseStrength = max(dot(norm, lightDir), 0.0);\n"
"	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);\n"
"	float shadow = 0.0f;\n"
"//#if defined(SOFTSHADOWS) || defined(HARDSHADOWS)\n"
"//	for (int i = 0; i < 5; i++) {\n"
"//		if (ClipSpaceZ <= Light.cascadeEnds[i]) {\n"
"//			shadow = CalcShadow(i, FragPosLightSpace[i]);\n"
"//			break;\n"
"//		}\n"
"//	}\n"
"//#endif\n"
"	vec3 ambient = Light.color * material.ambient * ambientStrength;\n"
"	vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;\n"
"	vec3 specular = (1.0f - shadow) * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;\n"
"	return (ambient + diffuse + specular);\n"
"}\n"
"vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)\n"
"{\n"
"	vec3 lightDir = normalize(Light.position - fragPos);\n"
"	vec3 reflectDir = reflect(-lightDir, normal);\n"
"	float ambientStrength = 1.0f;\n"
"	float diffuseStrength = max(dot(normal, lightDir), 0.0);\n"
"	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);\n"
"	float theta = dot(lightDir, normalize(-Light.direction));\n"
"	float epsilon = Light.innerSpotAngle - Light.spotAngle;\n"
"	float intensity = clamp((theta - Light.spotAngle) / epsilon, 0.0f, 1.0f);\n"
"	float shadow = CalcShadow(0, FragPosLightSpace[0]);\n"
"	//float attenuation = 1.0f / (1.0f + 0.01f*pow(length(Light.position - fragPos), 2));\n"
"	float distance = length(Light.position - fragPos);\n"
"	float attenuation = 1.0f;// 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);\n"
"	vec3 ambient = Light.color * material.ambient * ambientStrength;\n"
"	vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;\n"
"	vec3 specular = (1.0f - shadow) * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;\n"
"	ambient *= attenuation;\n"
"	diffuse *= attenuation * intensity;\n"
"	specular *= attenuation * intensity;\n"
"	return vec3(ambient + diffuse + specular);\n"
"}\n"
"vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)\n"
"{\n"
"	vec3 lightDir = normalize(Light.position - fragPos);\n"
"	vec3 reflectDir = reflect(-lightDir, normal);\n"
"	float ambientStrength = 1.0f;\n"
"	float diffuseStrength = max(dot(normal, lightDir), 0.0);\n"
"	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);\n"
"	float distance = length(Light.position - fragPos);\n"
"	float attenuation = 1.0f;// 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);\n"
"	vec3 ambient = Light.color * material.ambient * ambientStrength;\n"
"	vec3 diffuse = material.diffuse * diffuseStrength;\n"
"	vec3 specular = material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;\n"
"	ambient *= attenuation;\n"
"	diffuse *= attenuation;\n"
"	specular *= attenuation;\n"
"	return vec3(ambient + diffuse + specular);\n"
"}\n"
"float CalcShadow(int index, vec4 fragPosLightSpace)\n"
"{\n"
"	// only actually needed when using perspective projection for the light\n"
"	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;\n"
"	// projCoord is in [-1,1] range. Convert it ot [0,1] range.\n"
"	projCoords = projCoords * 0.5f + 0.5f;\n"
"	float closestDepth = texture(shadowMap[index], projCoords.xy).r;\n"
"	// get depth of current fragment from light's perspective\n"
"	float currentDepth = projCoords.z - 0.005f;\n"
"	// check whether current frag pos is in shadow\n"
"	float shadow = closestDepth < currentDepth ? 1.0f : 0.0f;\n"
"	return shadow;\n"
"}\n";