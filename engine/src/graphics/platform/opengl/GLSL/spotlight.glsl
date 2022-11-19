#ifndef SPOTLIGHT_GLSL__
#define SPOTLIGHT_GLSL__

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

#endif