#vertex
// INPUT/OUTPUT structures
struct VS_INPUT
{
    float2 position : POSITION;
    float2 texCoord : TEXCOORD;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = float4(input.position.x, input.position.y, 0.0f, 1.0f);
    output.texCoord = input.texCoord;
    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D texture0;
SamplerState sampleState
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(0.0f, 0.0f, texture0.Sample(sampleState, input.texCoord).b, 1.0f);
}