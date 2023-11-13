// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
};

// uniforms : external parameters
matrix worldViewProjection;

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(worldViewProjection, float4(input.position, 1.0));
    return output;
}