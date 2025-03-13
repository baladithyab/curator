# AWS Bedrock Integration for Curator

This document provides information on using AWS Bedrock with the Curator library.

## Overview

AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies through a unified API. Curator now supports AWS Bedrock for both online (real-time) and batch processing. This integration allows you to use models from providers such as Anthropic (Claude), Amazon (Titan/Nova), Meta (Llama), Mistral AI, and more through a single consistent interface.

## Features

- **Online Inference**: Real-time text generation through AWS Bedrock API
- **Batch Processing**: Efficient processing of large request volumes at 50% of on-demand pricing
- **Multi-Provider Support**: Compatible with all major model providers on AWS Bedrock
- **Inference Profiles**: Support for cross-region inference through AWS inference profiles
- **Automatic Model Detection**: Built-in logic to detect and use AWS Bedrock models based on model ID
- **Consistent API**: Use the same familiar Curator API for interacting with AWS Bedrock models

## Supported Models

### Online Inference

All AWS Bedrock models are supported for online inference, including:

- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku, Claude 3.5, etc.
- **Amazon**: Titan Text, Titan Embeddings, Nova, etc.
- **Meta**: Llama 3, Llama 3.2, Llama 3.3, etc.
- **Cohere**: Command, Command R, Embed, etc.
- **AI21**: Jurassic, Jamba, etc.
- **Mistral**: Various Mistral AI models

### Batch Processing

According to AWS documentation, batch inference is supported for models from four main providers at a 50% discount compared to on-demand pricing:

- **Anthropic**: 
  - Claude 3 Haiku (available in multiple regions including us-east-1, us-west-2)
  - Claude 3 Sonnet (available in multiple regions including us-east-1, us-west-2)
  - Claude 3 Opus (available in us-west-2)
  - Claude 3.5 Sonnet (available in multiple regions including us-east-1, us-west-2)
  - Claude 3.5 Haiku (available in us-west-2)
  - Claude 3.5 Sonnet v2 (available in us-west-2)

- **Amazon**: 
  - Titan Multimodal Embeddings G1 (available in multiple regions)
  - Titan Text Embeddings V2 (available in multiple regions)
  - Nova series (Lite, Micro, Pro) (available in us-east-1)

- **Meta**: 
  - Llama 3.1 series (8B, 70B, 405B) (primarily in us-west-2)
  - Llama 3.2 series (1B, 3B, 11B, 90B) (primarily in us-west-2 with some in eu-west-3)
  - Llama 3.3 70B Instruct (available in us-east-1, us-west-2)

- **Mistral AI**: 
  - Mistral Small (24.02) (available in us-east-1)
  - Mistral Large (24.07) (available in us-west-2)

Cohere and AI21 models are not officially listed by AWS as supporting batch inference. Regional availability may change over time, so always refer to the [official AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html) for the most up-to-date information.

For a detailed breakdown of model availability by region and model IDs, see our [AWS Bedrock Batch Model Availability](aws_bedrock_batch_model_availability.md) guide.

For information on input and output formats for batch inference, see our [AWS Bedrock Batch Input/Output Formats](aws_bedrock_batch_input_output.md) guide.

## Requirements

- AWS credentials (access key and secret key)
- Proper IAM permissions for AWS Bedrock
- For batch processing:
  - S3 bucket for input/output files
  - IAM role with permissions for both Bedrock and the S3 bucket

## Configuration

### Environment Variables

You can configure the AWS Bedrock integration using the following environment variables:

```bash
# Core AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1  # Default region

# For batch processing
export BEDROCK_BATCH_S3_BUCKET=your-s3-bucket
export BEDROCK_BATCH_S3_PREFIX=your-prefix  # Optional, defaults to "bedrock-batch"
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::123456789012:role/YourRoleName

# For inference profiles
export BEDROCK_USE_INFERENCE_PROFILE=true  # Optional, enables inference profiles
```

### Direct Configuration

You can also provide configuration directly when creating the processor:

```python
processor = curator.get_request_processor(
    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    backend="bedrock",  # Explicitly specify bedrock backend
    region_name="us-east-1",  # Optional AWS region
    use_inference_profile=True,  # Optional, use inference profiles
    # For batch processing
    batch=True,  # Enable batch mode
    s3_bucket="your-s3-bucket",  # S3 bucket for batch
    s3_prefix="your-prefix",  # Optional prefix for S3 objects
    role_arn="arn:aws:iam::123456789012:role/YourRoleName",  # IAM role for batch
)
```

## AWS Bedrock Model IDs

AWS Bedrock models use the following naming convention:

```
provider.model-name-version[:tag]
```

For example:
- `anthropic.claude-3-haiku-20240307-v1:0`
- `amazon.titan-text-express-v1`
- `meta.llama3-1-8b-instruct-v1:0`

## Inference Profiles

AWS Bedrock Inference Profiles allow for optimized cross-region performance. Inference profiles have the following format:

```
us.provider.model-name-version[:tag]
```

For example:
- `us.anthropic.claude-3-haiku-20240307-v1:0`
- `us.amazon.nova-lite-v1:0`
- `us.meta.llama3-1-8b-instruct-v1:0`

You can use inference profiles in two ways:

1. **Direct specification**:
   ```python
   processor = curator.get_request_processor(
       model_name="us.anthropic.claude-3-haiku-20240307-v1:0",
       backend="bedrock"
   )
   ```

2. **Automatic conversion**:
   ```python
   processor = curator.get_request_processor(
       model_name="anthropic.claude-3-haiku-20240307-v1:0",
       backend="bedrock",
       use_inference_profile=True  # Will convert to us.anthropic.claude-3-haiku-20240307-v1:0
   )
   ```

## Usage Examples

### Online Inference

```python
import os
from bespokelabs import curator

# Set AWS region if needed
os.environ["AWS_REGION"] = "us-east-1"

# Get processor for a Claude model on Bedrock
processor = curator.get_request_processor(
    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    generation_params={"temperature": 0.7, "max_tokens": 500}
)

# Generate response
response = processor.generate("Write a poem about cloud computing.")
print(response)
```

### Batch Processing

```python
import os
from bespokelabs import curator

# Set required environment variables
os.environ["AWS_REGION"] = "us-east-1"
os.environ["BEDROCK_BATCH_S3_BUCKET"] = "your-s3-bucket"
os.environ["BEDROCK_BATCH_ROLE_ARN"] = "arn:aws:iam::123456789012:role/BedrockBatchRole"

# Get batch processor
processor = curator.get_request_processor(
    model_name="anthropic.claude-3-haiku-20240307-v1:0",
    batch=True,
    generation_params={"temperature": 0.7, "max_tokens": 300}
)

# Create request in JSONL format
# Each line should be a GenericRequest in JSON format
with open("requests.jsonl", "w") as f:
    for i in range(5):
        request = curator.GenericRequest(
            task_id=i,
            prompt=f"Write about topic {i+1}",
            generation_params={"temperature": 0.7, "max_tokens": 300}
        )
        f.write(curator.json.dumps(request.to_dict()) + "\n")

# Process requests from files
processor.requests_to_responses(["requests.jsonl"])

# Results will be written to requests_responses.jsonl
```

### Using Inference Profiles

```python
# Direct inference profile usage
processor = curator.get_request_processor(
    model_name="us.anthropic.claude-3-haiku-20240307-v1:0",  # Note the 'us.' prefix
    backend="bedrock",
    generation_params={"temperature": 0.7, "max_tokens": 500}
)

# Auto-conversion to inference profile
processor = curator.get_request_processor(
    model_name="anthropic.claude-3-haiku-20240307-v1:0",  # Standard model ID
    backend="bedrock",
    use_inference_profile=True,  # This enables conversion
    generation_params={"temperature": 0.7, "max_tokens": 500}
)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure AWS credentials are properly set
   - Verify IAM permissions include the required Bedrock actions

2. **Model Not Available**
   - Check if your AWS account has access to the requested model
   - Verify the model is available in your specified region

3. **Batch Processing Failures**
   - Ensure the S3 bucket exists and is accessible
   - Verify the IAM role has proper permissions for both Bedrock and S3
   - Check if the model supports batch processing

4. **Inference Profile Issues**
   - Ensure your account has access to inference profiles
   - Verify the region supports the requested inference profile

### IAM Role for Batch Processing

The IAM role for batch processing needs the following permissions:

- `bedrock:CreateModelInvocationJob`
- `bedrock:GetModelInvocationJob`
- `bedrock:StopModelInvocationJob`
- `s3:GetObject` (on input S3 location)
- `s3:PutObject` (on output S3 location)
- `s3:ListBucket` (on S3 bucket)

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html)
- [AWS IAM Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)
- [AWS Bedrock Capabilities](aws_bedrock_capabilities.md)
- [AWS Bedrock Service Quotas](aws_bedrock_service_quotas.md)
- [AWS Bedrock Batch Model Availability](aws_bedrock_batch_model_availability.md) 