#version 330 core

#define NUM_OF_CASCADES 5

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
};

uniform Material material;

uniform sampler2D mainTexture;
uniform sampler2D shadowMap[NUM_OF_CASCADES];
uniform samplerCube cubeShadowMap;

//uniform float farPlane;

in float FarPlane;
in float ClipSpaceZ;
in float CascadeEnds[NUM_OF_CASCADES];
in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace[NUM_OF_CASCADES];

out vec4 FragColor;

vec2 poissonDisk[4] = vec2[](
  vec2( -0.94201624, -0.39906216 ),
  vec2( 0.94558609, -0.76890725 ),
  vec2( -0.094184101, -0.92938870 ),
  vec2( 0.34495938, 0.29387760 )
);

float CalcShadow(int index, vec4 FragPosLightSpace);
float CalcPointShadow(vec3 fragPos, vec3 lightPos);

vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);
vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);


void main(void) 
{
	vec3 viewDir = normalize(CameraPos - FragPos);

	//vec3 finalLight = FragPos - PointLight.position;

	vec3 finalLight = vec3(0.05f, 0.05f, 0.05f);
	
	finalLight += CalcDirLight(material, Normal, viewDir);
	//finalLight += CalcSpotLight(material, Normal, FragPos, viewDir);
	//finalLight += CalcPointLight(material, Normal, FragPos, viewDir);

	FragColor = vec3(1.0f, 0.0f, 0.0f, 1.0f);//texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f);

	/*if(ClipSpaceZ <= cascadeEnds[0]){
		FragColor = texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f) * vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else if(ClipSpaceZ <= cascadeEnds[1]){
		FragColor = texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f) * vec4(1.0f, 1.0f, 0.0f, 1.0f);
	}
	else if(ClipSpaceZ <= cascadeEnds[2]){
		FragColor = texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f) * vec4(0.0f, 1.0f, 0.0f, 1.0f);
	}
	else if(ClipSpaceZ <= cascadeEnds[3]){
		FragColor = texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f) * vec4(0.0f, 0.0f, 1.0f, 1.0f);
	}
	else if(ClipSpaceZ <= cascadeEnds[4]){
		FragColor = texture(mainTexture, TexCoord) * vec4(finalLight, 1.0f) * vec4(0.6f, 0.0f, 0.6f, 1.0f);
	}
	else{
		FragColor = vec4(0.5, 0.5, 0.5, 1.0);
	}*/
}


float CalcShadow(int index, vec4 fragPosLightSpace)
{
	// only actually needed when using perspective projection for the light
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

  	// projCoord is in [-1,1] range. Convert it ot [0,1] range.
	projCoords = projCoords * 0.5f + 0.5f;

	float bias = 0.005f;
	float shadow = 0.0f;
	for(int i=0;i<4;i++){
		if(texture(shadowMap[index], projCoords.xy + poissonDisk[i]/700.0).r < projCoords.z - bias){
			shadow += 0.2f;
		}
	}

	return shadow;
}

float CalcPointShadow(vec3 fragPos, vec3 lightPos)
{
	// get vector between fragment position and light position
    vec3 fragToLight = fragPos - lightPos;  
    float closestDepth = texture(cubeShadowMap, fragToLight).x;

    //return closestDepth;

    // it is currently in linear range between [0,1]. Re-transform back to original value
    closestDepth *= FarPlane;

    // now get current linear depth as the length between the fragment and light position
    float currentDepth = length(fragToLight);

    // now test for shadows
    float bias = 0.05; 
    float shadow = currentDepth -  bias > closestDepth ? 1.0 : 0.0;

    return shadow;
}

vec3 helper(vec3 fragPos, vec3 lightPos)
{
	vec3 fragToLight = fragPos - lightPos;  
    float closestDepth = texture(cubeShadowMap, fragToLight).x;
	return vec3(closestDepth, closestDepth, closestDepth);
}


vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)
{
	vec3 lightDir = normalize(-DirectionalLight.direction);

	vec3 reflectDir = reflect(-lightDir, normal);
    
    float ambientStrength = 1.0f;
    float diffuseStrength = max(dot(normal, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

	float shadow = 0.0f;
	for(int i = 0; i < NUM_OF_CASCADES; i++){
		if(ClipSpaceZ <= CascadeEnds[i]){
			shadow = CalcShadow(i, FragPosLightSpace[i]);
			break;
		}
	}

	vec3 ambient = DirectionalLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = (1.0f - shadow) * DirectionalLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = (1.0f - shadow) * DirectionalLight.specular * material.specular * specularStrength;

    return (ambient + diffuse + specular);
}

vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(SpotLight.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 0.05f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float attenuation = 1.0f / (1.0f + 0.01f*pow(length(SpotLight.position - fragPos), 2));

	float theta = dot(lightDir, normalize(-SpotLight.direction));
	float epsilon = SpotLight.cutOff - SpotLight.outerCutOff;
	float intensity = clamp((theta - SpotLight.outerCutOff) / epsilon, 0.0f, 1.0f);

	float shadow = CalcShadow(0, FragPosLightSpace[0]);

	vec3 ambient = SpotLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = (1.0f - shadow) * SpotLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = (1.0f - shadow) * SpotLight.specular * material.specular * specularStrength;

	ambient *= attenuation * intensity;
	diffuse *= attenuation * intensity;
	specular *= attenuation * intensity;

	return vec3(ambient + diffuse + specular);
}

vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(PointLight.position - fragPos);

	vec3 reflectDir = reflect(-lightDir, normal);

	float ambientStrength = 0.05f;
	float diffuseStrength = max(dot(normal, lightDir), 0.0);
	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);

	float distance = length(PointLight.position - fragPos);
	float attenuation = 1.0f;// / (PointLight.constant + PointLight.linear * distance + PointLight.quadratic * distance * distance);

	//float shadow = CalcPointShadow(fragPos, PointLight.position);

	//return vec3(shadow, shadow, shadow);

	//return helper(fragPos, PointLight.position);
	float shadow = 1.0f - helper(fragPos, PointLight.position).x;


	vec3 ambient = PointLight.ambient * material.ambient * ambientStrength;
	vec3 diffuse = (1.0f - shadow) * PointLight.diffuse * material.diffuse * diffuseStrength;
	vec3 specular = (1.0f - shadow) * PointLight.specular * material.specular * specularStrength;

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	return vec3(ambient + diffuse + specular);
}