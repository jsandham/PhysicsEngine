#vertex
// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

// uniforms : external parameters
matrix worldViewProjection;

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(worldViewProjection, float4(input.position, 1.0));
    output.normal = input.normal;
    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(input.normal, 1.0);
}