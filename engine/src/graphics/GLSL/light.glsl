layout(std140) uniform LightBlock
{
    mat4 lightProjection[5]; // 0    64   128  192  256
    mat4 lightView[5]; // 320  384  448  512  576
    vec3 position; // 640
    vec3 direction; // 656
    vec3 color; // 672
    float cascadeEnds[5]; // 688  704  720  736  752
    float intensity; // 768
    float spotAngle; // 772
    float innerSpotAngle; // 776
    float shadowNearPlane; // 780
    float shadowFarPlane; // 784
    float shadowBias; // 788
    float shadowRadius; // 792
    float shadowStrength; // 796
}Light;