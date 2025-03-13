# AWS Bedrock Batch Inference - Model Availability

This document provides detailed information about which AWS Bedrock models support batch inference and in which regions they are available. The information is based on the [official AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html).

## Model Availability by Provider and Region

AWS Bedrock batch inference offers a 50% discount compared to on-demand pricing for the following models:

### Amazon Models

| Model | Foundation Model Regions | Inference Profile Regions | Custom Model |
|-------|--------------------------|---------------------------|-------------|
| Nova Lite | us-east-1 | us-east-1, us-east-2, us-west-2 | N/A |
| Nova Micro | us-east-1 | us-east-1, us-east-2, us-west-2 | N/A |
| Nova Pro | us-east-1 | us-east-1, us-east-2, us-west-2 | N/A |
| Titan Multimodal Embeddings G1 | us-east-1, us-west-2, ap-northeast-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1 | N/A | us-east-1, us-west-2 |
| Titan Text Embeddings V2 | us-east-1, us-west-2, ap-northeast-2, ca-central-1, eu-central-1, eu-west-2, sa-east-1 | N/A | N/A |

### Anthropic Models

| Model | Foundation Model Regions | Inference Profile Regions | Custom Model |
|-------|--------------------------|---------------------------|-------------|
| Claude 3.5 Haiku | us-west-2 | us-east-1, us-west-2 | N/A |
| Claude 3.5 Sonnet | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-central-1 | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, eu-central-1, eu-west-1, eu-west-3 | N/A |
| Claude 3.5 Sonnet v2 | us-west-2 | us-east-1, us-east-2, us-west-2 | N/A |
| Claude 3 Haiku | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1 | N/A | N/A |
| Claude 3 Opus | us-west-2 | us-east-1, us-west-2 | N/A |
| Claude 3 Sonnet | us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1 | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1 | N/A |

### Meta Models

| Model | Foundation Model Regions | Inference Profile Regions | Custom Model |
|-------|--------------------------|---------------------------|-------------|
| Llama 3.1 405B Instruct | us-west-2 | N/A | N/A |
| Llama 3.1 70B Instruct | us-west-2 | us-east-1, us-west-2 | N/A |
| Llama 3.1 8B Instruct | us-west-2 | us-east-1, us-west-2 | N/A |
| Llama 3.2 11B Instruct | us-west-2 | us-east-1, us-west-2 | N/A |
| Llama 3.2 1B Instruct | us-west-2, eu-west-3 | us-east-1, us-west-2, eu-central-1, eu-west-1, eu-west-3 | N/A |
| Llama 3.2 3B Instruct | us-west-2, eu-west-3 | us-east-1, us-west-2, eu-central-1, eu-west-1, eu-west-3 | N/A |
| Llama 3.2 90B Instruct | us-west-2 | us-east-1, us-west-2 | N/A |
| Llama 3.3 70B Instruct | us-east-1, us-west-2 | us-east-1, us-east-2, us-west-2 | N/A |

### Mistral AI Models

| Model | Foundation Model Regions | Inference Profile Regions | Custom Model |
|-------|--------------------------|---------------------------|-------------|
| Mistral Large (24.07) | us-west-2 | N/A | N/A |
| Mistral Small (24.02) | us-east-1 | N/A | N/A |

## Understanding Regions and Profiles

### Foundation Model Regions
These are the regions where the model is directly available and can be used for batch inference.

### Inference Profile Regions
These regions can access the model through cross-region inference profiles, which allows you to use models that might not be directly available in your primary region.

### Cross-Region Inference Profiles

AWS Bedrock provides system-defined inference profiles that allow models to be used across multiple regions. For batch inference, these profiles have the format:

```
us.provider.model-name-version[:tag]
```

For example:
- `us.anthropic.claude-3-haiku-20240307-v1:0`
- `us.meta.llama3-1-8b-instruct-v1:0`

## Model IDs for Batch Support

For the Curator library, the following model IDs are officially documented to support batch inference:

### Anthropic
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-7-sonnet-20250219-v1:0`

### Amazon
- `amazon.titan-text-express-v1`
- `amazon.titan-text-lite-v1`
- `amazon.titan-embed-text-v1`

### Meta
- `meta.llama3-1-8b-instruct-v1:0`
- `meta.llama3-1-70b-instruct-v1:0`
- `meta.llama3-1-405b-instruct-v1:0`
- `meta.llama3-2-1b-instruct-v1:0`
- `meta.llama3-2-3b-instruct-v1:0`
- `meta.llama3-2-11b-instruct-v1:0`
- `meta.llama3-2-90b-instruct-v1:0`
- `meta.llama3-3-70b-instruct-v1:0`

### Mistral AI
- `mistral.mistral-large-2402-v1:0` (Mistral Small 24.02)
- `mistral.mistral-large-2407-v1:0` (Mistral Large 24.07)

## Notes and Best Practices

1. **Region Considerations**: Always consider the region where your batch job will run. Some models are only available in specific regions.

2. **Inference Profiles**: When a model is not available in your preferred region, consider using an inference profile to access it from a supported region.

3. **Cost Efficiency**: Batch inference offers a 50% discount compared to on-demand pricing, making it cost-effective for large-scale processing.

4. **Staying Updated**: AWS regularly adds new models and regions for batch inference. Check the official documentation for the most current information.

## Reference

For the most up-to-date information on AWS Bedrock batch inference model support and regional availability, refer to the [official AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html).

## Related Documentation

- [AWS Bedrock Integration](aws_bedrock_integration.md)
- [AWS Bedrock Capabilities](aws_bedrock_capabilities.md)
- [AWS Bedrock Service Quotas](aws_bedrock_service_quotas.md)
- [AWS Bedrock Batch Input/Output Formats](aws_bedrock_batch_input_output.md) 