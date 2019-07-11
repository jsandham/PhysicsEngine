VERTEX:

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}Camera;

layout (std140) uniform LightBlock
{
	mat4 lightProjection[5]; // 0    64   128  192  256
	mat4 lightView[5];       // 320  384  448  512  576 
	vec3 position;           // 640
	vec3 direction;          // 656
	vec3 ambient;            // 672
	vec3 diffuse;            // 688
	vec3 specular;           // 704
	float cascadeEnds[5];    // 720  736  752  768  784
	float farPlane;          // 800
	float constant;          // 804
	float linear;            // 808
	float quadratic;         // 812
	float cutOff;            // 816
	float outerCutOff;       // 820
}Light;

uniform mat4 model;

in vec3 position;
in vec3 normal;
in vec2 texCoord;

out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec2 TexCoord;

out float ClipSpaceZ;
out vec4 FragPosLightSpace[5];

void main()
{
    CameraPos = Camera.cameraPos;
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;  
    TexCoord = texCoord;
    
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);

	ClipSpaceZ = gl_Position.z;
	for(int i = 0; i < 5; i++){
		FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);
	}
}





FRAGMENT:

layout (std140) uniform LightBlock
{
	mat4 lightProjection[5]; // 0    64   128  192  256
	mat4 lightView[5];       // 320  384  448  512  576 
	vec3 position;           // 640
	vec3 direction;          // 656
	vec3 ambient;            // 672
	vec3 diffuse;            // 688
	vec3 specular;           // 704
	float cascadeEnds[5];    // 720  736  752  768  784
	float farPlane;          // 800
	float constant;          // 804
	float linear;            // 808
	float quadratic;         // 812
	float cutOff;            // 816
	float outerCutOff;       // 820
}Light;

struct Material
{
	float shininess;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	sampler2D mainTexture;
	sampler2D normalMap;
	sampler2D specularMap;
};

uniform Material material;
uniform sampler2D shadowMap[5];

in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;

in float ClipSpaceZ;
in vec4 FragPosLightSpace[5];

vec2 poissonDisk[4] = vec2[](
  vec2( -0.94201624, -0.39906216 ),
  vec2( 0.94558609, -0.76890725 ),
  vec2( -0.094184101, -0.92938870 ),
  vec2( 0.34495938, 0.29387760 )
);





out vec4 FragColor;

vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);
vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);

float CalcShadow(int index, vec4 fragPosLightSpace);

void main(void) 
{
	vec3 viewDir = normalize(CameraPos - FragPos);

#if defined(DIRECTIONALLIGHT)
	//FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);
	if(ClipSpaceZ <= Light.cascadeEnds[0]){
		FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord) * vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else if(ClipSpaceZ <= Light.cascadeEnds[1]){
		FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord) * vec4(0.0f, 1.0f, 0.0f, 1.0f);
	}
	else if(ClipSpaceZ <= Light.cascadeEnds[2]){
		FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord) * vec4(0.0f, 0.0f, 1.0f, 1.0f);
	}
	else if(ClipSpaceZ <= Light.cascadeEnds[3]){
		FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord) * vec4(0.0f, 1.0f, 1.0f, 1.0f);
	}
	else if(ClipSpaceZ <= Light.cascadeEnds[4]){
		FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord) * vec4(0.6f, 0.0f, 0.6f, 1.0f);
	}
	else{
		FragColor = vec4(0.5, 0.5, 0.5, 1.0);
	}
#elif defined(SPOTLIGHT)
	FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);
#elif defined(POINTLIGHT)
	FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);
#else 
	FragColor = vec4(0.5, 0.5, 0.5, 1.0);
#endif
}

vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)
{
	vec3 norm = normalize(normal);
	vec3 lightDir = normalize(Light.direction);

	vec3 reflectDir = reflect(-lightDir, norm);
    
    float ambientStrength = 1.0f;
    float diffuseStrength = max(dot(norm, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    float shadow = 0.0f;
	for(int i = 0; i < 5; i++){
		if(ClipSpaceZ <= Light.cascadeEnds[i]){
			shadow = CalcShadow(i, FragPosLightSpace[i]);
			break;
		}
	}

	vec3 ambient = Light.ambient * material.ambient * ambientStrength;
	vec3 diffuse = (1.0f - shadow) * Light.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = (1.0f - shadow) * Light.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;

    return (ambient + diffuse + specular);
}

vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(Light.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 1.0f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float theta = dot(lightDir, normalize(-Light.direction));
	float epsilon = Light.cutOff - Light.outerCutOff;
	float intensity = clamp((theta - Light.outerCutOff) / epsilon, 0.0f, 1.0f);

	float shadow = CalcShadow(0, FragPosLightSpace[0]);

	//float attenuation = 1.0f / (1.0f + 0.01f*pow(length(Light.position - fragPos), 2));
	float distance = length(Light.position - fragPos);
	float attenuation = 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);

	vec3 ambient = Light.ambient * material.ambient * ambientStrength;
	vec3 diffuse = (1.0f - shadow) * Light.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = (1.0f - shadow) * Light.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;

	ambient *= attenuation;
	diffuse *= attenuation * intensity;
	specular *= attenuation * intensity;

	// return vec3(FragPosLightSpace[0].x/FragPosLightSpace[0].w, FragPosLightSpace[0].y/FragPosLightSpace[0].w, FragPosLightSpace[0].z/FragPosLightSpace[0].w);//vec3(ambient + diffuse + specular);
	// return vec3(0.0f, 0.0f, FragPosLightSpace[0].z/FragPosLightSpace[0].w);
	return vec3(ambient + diffuse + specular);
}

vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(Light.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 1.0f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float distance = length(Light.position - fragPos);
	float attenuation = 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);

	vec3 ambient = Light.ambient * material.ambient * ambientStrength;
	vec3 diffuse = Light.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = Light.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;
	
	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return vec3(ambient + diffuse + specular);
}


float CalcShadow(int index, vec4 fragPosLightSpace)
{
	// only actually needed when using perspective projection for the light
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

  	// projCoord is in [-1,1] range. Convert it ot [0,1] range.
	projCoords = projCoords * 0.5f + 0.5f;

	float closestDepth = texture(shadowMap[index], projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z - 0.005f;
    // check whether current frag pos is in shadow
    float shadow = closestDepth < currentDepth ? 1.0f : 0.0f;

    //float z = closestDepth * 2.0 - 1.0; // back to NDC 
    //return (2.0 * 0.1f * 250.0f) / (250.0f + 0.1f - z * (250.0f - 0.1f));

    return shadow;
    // return projCoords.z;// > 2000000.0f ? 1.0f : 0.0f;

	// float bias = 0.005f;
	// float shadow = 0.0f;
	// for(int i = 0; i < 4; i++){
	// 	if(texture(shadowMap[index], projCoords.xy + poissonDisk[i] / 700.0f).r < projCoords.z - bias){
	// 		shadow += 0.2f;
	// 	}
	// }

	// return shadow;
}