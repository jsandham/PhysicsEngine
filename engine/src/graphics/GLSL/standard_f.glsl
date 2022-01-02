#version 430 core
#include "light.glsl"
#include "material.glsl"

uniform Material material;
uniform sampler2D shadowMap[5];
in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;
in float ClipSpaceZ;
in vec4 FragPosLightSpace[5];
vec2 poissonDisk[4] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.094184101, -0.92938870), vec2(0.34495938, 0.29387760));
out vec4 FragColor;
vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);
vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);
float CalcShadow(int index, vec4 fragPosLightSpace);
void main(void)
{
    vec3 viewDir = normalize(CameraPos - FragPos);
    vec4 albedo = vec4(material.colour, 1.0);
    if(material.sampleMainTexture == 1)
    {
        albedo = texture(material.mainTexture, TexCoord);
    }
#if defined (DIRECTIONALLIGHT)
    FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * albedo;
#elif defined(SPOTLIGHT)
    FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;
#elif defined(POINTLIGHT)
    FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;
#else
    FragColor = vec4(0.5, 0.5, 0.5, 1.0) * albedo;
#endif
}
vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-Light.direction);
    vec3 reflectDir = reflect(-lightDir, norm);
    float diffuseStrength = max(dot(norm, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    float shadow = 0.0f;
#if defined (SOFTSHADOWS) || defined(HARDSHADOWS)
    for (int i = 0; i < 5; i++)
    {
        if(ClipSpaceZ <= Light.cascadeEnds[i])
        {
            shadow = Light.shadowStrength * CalcShadow(i, FragPosLightSpace[i]);
            break;
        }
    }
#endif
    vec3 ambient = material.ambient;
    vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;
    vec3 specular = (1.0f - shadow) * material.specular * specularStrength;
    diffuse = diffuse * Light.intensity * Light.color;
    specular = specular * Light.intensity * Light.color;
    vec3 finalColor = (ambient + diffuse + specular);
#if defined (SHOWCASCADES)
    if(ClipSpaceZ <= Light.cascadeEnds[0])
    {
        finalColor = finalColor * vec3(1.0f, 0.0f, 0.0f);
    }
    else if (ClipSpaceZ <= Light.cascadeEnds[1])
    {
        finalColor = finalColor * vec3(0.0f, 1.0f, 0.0f);
    }
    else if (ClipSpaceZ <= Light.cascadeEnds[2])
    {
        finalColor = finalColor * vec3(0.0f, 0.0f, 1.0f);
    }
    else if (ClipSpaceZ <= Light.cascadeEnds[3])
    {
        finalColor = finalColor * vec3(0.0f, 1.0f, 1.0f);
    }
    else if (ClipSpaceZ <= Light.cascadeEnds[4])
    {
        finalColor = finalColor * vec3(0.6f, 0.0f, 0.6f);
    }
    else
    {
        finalColor = vec3(0.5, 0.5, 0.5);
    }
#endif
return finalColor;
}
vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(Light.position - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float diffuseStrength = max(dot(normal, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);
    float theta = dot(lightDir, normalize(-Light.direction));
    float epsilon = Light.innerSpotAngle - Light.spotAngle;
    float intensity = clamp((theta - Light.spotAngle) / epsilon, 0.0f, 1.0f);
    float shadow = 0;
#if defined (SOFTSHADOWS) || defined(HARDSHADOWS)
    shadow = Light.shadowStrength * CalcShadow(0, FragPosLightSpace[0]);
#endif
    float distance = length(Light.position - fragPos);
    float attenuation = 1.0f; // / (1.0f + 0.0f * distance + 0.01f * distance * distance);
    vec3 ambient = material.ambient;
    vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;
    vec3 specular = (1.0f - shadow) * material.specular * specularStrength;
    ambient *= attenuation;
    diffuse *= attenuation * intensity * Light.intensity * Light.color;
    specular *= attenuation * intensity * Light.intensity * Light.color;
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
    float attenuation = 1.0f; // 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);
    vec3 ambient = material.ambient * ambientStrength;
    vec3 diffuse = material.diffuse * diffuseStrength;
    vec3 specular = material.specular * specularStrength;
    // vec3 specular = material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;
    ambient *= attenuation;
    diffuse *= attenuation * Light.intensity * Light.color;
    specular *= attenuation * Light.intensity * Light.color;
    return vec3(ambient + diffuse + specular);
}
float CalcShadow(int index, vec4 fragPosLightSpace)
{
    // only actually needed when using perspective projection for the light
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // projCoord is in [-1,1] range. Convert it ot [0,1] range.
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap[index], projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z; // - 0.005;
    // check whether current frag pos is in shadow
    // float shadow = closestDepth < currentDepth ? 1.0 : 0.0;
    // float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = currentDepth - Light.shadowBias > closestDepth ? 1.0 : 0.0;
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
};