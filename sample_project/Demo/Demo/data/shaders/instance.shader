VERTEX:

#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}Camera;

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in mat4 instanceModel;

out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec2 TexCoord;

void main()
{
    gl_Position = Camera.projection * Camera.view * instanceModel * vec4(position, 1.0); 

    FragPos = vec3(instanceModel * vec4(position, 1.0));
    CameraPos = Camera.cameraPos;
    Normal = mat3(transpose(inverse(instanceModel))) * normal;
    TexCoords = texCoord;
}


FRAGMENT:

#version 330 core

layout (std140) uniform DirectionalLightBlock
{
	vec3 direction;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
}DirectionalLight;

layout (std140) uniform SpotLightBlock
{
	vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular; 
  
    float constant;
    float linear;
    float quadratic;
    float cutOff;
    float outerCutOff;  
}SpotLight;

layout (std140) uniform PointLightBlock
{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	
	float constant;
	float linear;
	float quadratic;
}PointLight;

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

in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);
vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);


void main(void) 
{
	vec3 viewDir = normalize(CameraPos - FragPos);

	FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord); 
	//FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);
}

vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)
{
	vec3 lightDir = normalize(-DirectionalLight.direction);

	vec3 reflectDir = reflect(-lightDir, normal);
    
    float ambientStrength = 1.0f;
    float diffuseStrength = max(dot(normal, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

	vec3 ambient = DirectionalLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = DirectionalLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = DirectionalLight.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;

    return (ambient + diffuse + specular);
}

vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(SpotLight.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 1.0f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = 1.0f;//pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float attenuation = 1.0f / (1.0f + 0.01f*pow(length(SpotLight.position - fragPos), 2));

	float theta = dot(lightDir, normalize(-SpotLight.direction));
	float epsilon = SpotLight.cutOff - SpotLight.outerCutOff;
	float intensity = clamp((theta - SpotLight.outerCutOff) / epsilon, 0.0f, 1.0f);

	vec3 ambient = SpotLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = SpotLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = SpotLight.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;

	ambient *= attenuation * intensity;
	diffuse *= attenuation * intensity;
	specular *= attenuation * intensity;

	return vec3(ambient + diffuse + specular);
}

vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(PointLight.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 1.0f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float distance = length(PointLight.position - fragPos);
	float attenuation = 1.0f / (PointLight.constant + PointLight.linear * distance + PointLight.quadratic * distance * distance);

	vec3 ambient = PointLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = PointLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = PointLight.specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;
	
	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return vec3(ambient + diffuse + specular);
}