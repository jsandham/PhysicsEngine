#ifndef POINTLIGHT_GLSL__
#define POINTLIGHT_GLSL__

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

#endif