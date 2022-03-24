#ifndef DIRECTIONALLIGHT_GLSL__
#define DIRECTIONALLIGHT_GLSL__

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

    // tint color for cascades
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

#endif