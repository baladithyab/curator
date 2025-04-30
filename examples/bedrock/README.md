# AWS Bedrock Integration

This directory contains comprehensive examples for the AWS Bedrock integration in the Curator project.

## Examples

- `bedrock_online_examples.py`: Demonstrates how to use Curator with AWS Bedrock for online inference with various models (Claude 3.5 v2, Llama 3.3, Nova Pro, etc.)
- `bedrock_batch_examples.py`: Demonstrates how to use Curator with AWS Bedrock for batch processing with Claude 3.5 Haiku and Nova Micro

## Utility Scripts

- `setup_bedrock_resources.py`: Helper script to create and manage AWS resources (S3 bucket, IAM role) needed for batch processing

## Running the Examples

### Prerequisites

To run the examples, you need:

1. AWS credentials with access to Bedrock
2. Access to the models used in the examples
3. For batch examples, an S3 bucket and IAM role with appropriate permissions

### Setting Up AWS Credentials

You can set up AWS credentials in several ways:

1. Environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1
   ```

2. AWS credentials file (`~/.aws/credentials`):
   ```
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   ```

3. AWS config file (`~/.aws/config`):
   ```
   [default]
   region = us-east-1
   ```

### Running the Online Examples

```bash
# Run all examples with Claude 3.5 Sonnet v2
python bedrock_online_examples.py --model anthropic.claude-3-5-sonnet-20241022-v2:0 --region us-west-2

# Run a specific example
python bedrock_online_examples.py --example simple --region us-west-2

# Run the model comparison example
python bedrock_online_examples.py --example comparison --region us-west-2
```

### Running the Batch Examples

First, you need to set up the required AWS resources (S3 bucket and IAM role):

```bash
# Create the necessary AWS resources
python setup_bedrock_resources.py --create

# List existing resources
python setup_bedrock_resources.py --list
```

Then run the batch examples:

```bash
# Run all examples with automatically created resources
python bedrock_batch_examples.py --create-resources --region us-west-2

# Run with existing resources
python bedrock_batch_examples.py --s3-bucket your-bucket --role-arn arn:aws:iam::123456789012:role/your-role --region us-west-2

# Run a specific example
python bedrock_batch_examples.py --example simple --s3-bucket your-bucket --role-arn arn:aws:iam::123456789012:role/your-role

# Clean up resources after running
python setup_bedrock_resources.py --destroy
```

## Environment Variables

The examples use the following environment variables:

- `AWS_REGION`: AWS region to use for Bedrock API calls (default: us-west-2)
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
- `AWS_SESSION_TOKEN`: Your AWS session token (if using temporary credentials)
- `BEDROCK_BATCH_S3_BUCKET`: S3 bucket to use for batch input/output
- `BEDROCK_BATCH_ROLE_ARN`: IAM role ARN with permissions for Bedrock batch jobs
- `CURATOR_LOG_LEVEL`: Set to "DEBUG" for more verbose logging output

## Models Used

The examples cover the following AWS Bedrock models:

- Claude 3.5 Sonnet v2 (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
- Claude 3.5 Haiku (`anthropic.claude-3-5-haiku-20241022-v1:0`)
- Llama 3.3 70B (`meta.llama3-3-70b-instruct-v1:0`)
- Amazon Nova Pro (`amazon.nova-pro-v1:0`)
- Amazon Nova Micro (`amazon.nova-micro-v1:0`)
- Mistral Large (`mistral.mistral-large-2407-v1:0`)

## Troubleshooting

### Common Issues

1. **Missing AWS Credentials**: Ensure that your AWS credentials are properly set up and have access to Bedrock.

2. **Model Access**: Make sure you have access to the models you're trying to use. You can check this in the AWS Bedrock console.

3. **S3 Bucket Permissions**: For batch processing, ensure that your IAM role has the necessary permissions to access the S3 bucket.

4. **Region Compatibility**: Make sure you're using a region where Bedrock is available and your models are accessible.

### Getting Help

If you encounter issues with the Bedrock integration, check the following resources:

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Curator Documentation](https://curator.readthedocs.io/)
- [AWS Bedrock Forum](https://forums.aws.amazon.com/)
