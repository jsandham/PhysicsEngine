struct PS_INPUT
{
    float4 position : SV_POSITION;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(1.0, 0.0, 0.0, 1.0);
}