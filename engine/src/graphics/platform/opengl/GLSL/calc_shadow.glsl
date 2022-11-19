#ifndef CALC_SHADOW_GLSL__
#define CALC_SHADOW_GLSL__

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

#endif