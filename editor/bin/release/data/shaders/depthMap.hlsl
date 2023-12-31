#vertex
// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
};

// constant buffers
cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
{
    matrix projection;
    matrix view;
    matrix viewProjection;
    float3 cameraPos;
}

cbuffer VS_CONSTANT_BUFFER2 : register(b3)
{
    matrix model;
}

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    float4x4 modelViewProj = mul(projection, mul(view, model));
    VS_OUTPUT output;
    output.position = mul(modelViewProj, float4(input.position, 1.0f));

    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}