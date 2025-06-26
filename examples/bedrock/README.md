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

# Wait for batch completion with progress reporting (timeout: 2 hours)
python bedrock_batch_examples.py --create-resources --wait --timeout 2

# Resume a previously submitted batch job
python bedrock_batch_examples.py --resume batch-123 --job-arn job-456 --wait --timeout 3

# Test timeout and wait functionality
python test_timeout_wait.py --s3-bucket your-bucket --role-arn arn:aws:iam::123456789012:role/your-role --wait --timeout 1

# Clean up resources after running
python setup_bedrock_resources.py --destroy
```

### Timeout and Wait Functionality

The Bedrock batch processor now supports configurable timeout settings and a wait option for synchronous batch processing:

1. **Timeout Control**: Set maximum wait time with `--timeout` parameter (in hours, default: 24, minimum: 24)
   - Note: AWS Bedrock requires a minimum timeout of 24 hours for batch jobs
   - If a lower value is specified, it will be automatically adjusted to 24 hours
2. **Synchronous Mode**: Use `--wait` flag to wait for batch completion with progress reporting
3. **Resume Functionality**: Use `--resume` with `--job-arn` to check or continue monitoring a previous job

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

## Advanced Features

### Timeout and Wait Functionality

The Bedrock batch processor supports configurable timeout settings and a wait option for synchronous batch processing.

#### Timeout Settings

The timeout setting controls how long the processor will wait for a batch job to complete before giving up. The default timeout is 24 hours, and AWS Bedrock requires a minimum timeout of 24 hours for batch jobs:

```python
# Set timeout in hours (minimum 24 hours for AWS Bedrock)
timeout_hours = 48

# Option 1: Set in BatchRequestProcessorConfig
config = BatchRequestProcessorConfig(
    model="anthropic.claude-3-5-haiku-20241022-v1:0",
    timeout_hours=timeout_hours
)

# Option 2: Set in backend_params
llm = curator.LLM(
    model_name="anthropic.claude-3-5-haiku-20241022-v1:0",
    backend="bedrock",
    batch=True,
    backend_params={
        "timeout_hours": timeout_hours
    }
)

# Option 3: Set directly on processor instance
processor = BedrockBatchRequestProcessor(...)
processor.timeout_hours = timeout_hours
```

> **Note**: If you specify a timeout less than 24 hours, it will be automatically adjusted to 24 hours when submitting the batch job to AWS Bedrock.

#### Wait Functionality

The wait functionality allows you to monitor a batch job until completion or timeout:

```python
# Submit a batch job
batch = await processor.submit_batch(batch)

# Wait for completion with progress reporting
batch = await processor.monitor_batch_job(
    batch,
    timeout_seconds=timeout_hours * 3600,
    verbose=True
)

# Check if the job completed successfully
if batch.status == GenericBatchStatus.COMPLETE:
    # Fetch the results
    responses = await processor.fetch_batch_results(batch)
```

#### Progress Reporting

The `monitor_batch_job` method provides real-time progress reporting:

```python
# Define a custom progress callback
def progress_callback(batch, elapsed_time, total_time, status_info):
    progress_pct = status_info["progress"] * 100
    status = status_info["bedrock_status"]
    print(f"Progress: {progress_pct:.1f}% | Status: {status}")

# Monitor with custom progress reporting
batch = await processor.monitor_batch_job(
    batch,
    timeout_seconds=timeout_hours * 3600,
    verbose=True,
    progress_callback=progress_callback
)
```

#### Resume Functionality

You can resume monitoring or retrieve results from a previously submitted batch job:

```python
# Create a minimal batch object with the job information
batch = GenericBatch(
    batch_id=batch_id,
    provider_batch_id=job_arn,
    status=GenericBatchStatus.PROCESSING
)

# Check current status
batch = await processor.retrieve_batch(batch)

# Resume monitoring if still processing
if batch.status == GenericBatchStatus.PROCESSING:
    batch = await processor.monitor_batch_job(batch)

# Fetch results if complete
if batch.status == GenericBatchStatus.COMPLETE:
    responses = await processor.fetch_batch_results(batch)
```

## Troubleshooting

### Common Issues

1. **Missing AWS Credentials**: Ensure that your AWS credentials are properly set up and have access to Bedrock.

2. **Model Access**: Make sure you have access to the models you're trying to use. You can check this in the AWS Bedrock console.

3. **S3 Bucket Permissions**: For batch processing, ensure that your IAM role has the necessary permissions to access the S3 bucket.

4. **Region Compatibility**: Make sure you're using a region where Bedrock is available and your models are accessible.

5. **Batch Job Timeouts**: If batch jobs are timing out, try increasing the timeout value with the `--timeout` parameter.

6. **Resume Issues**: When resuming a batch job, make sure to provide both the batch ID and job ARN.

### Getting Help

If you encounter issues with the Bedrock integration, check the following resources:

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Curator Documentation](https://curator.readthedocs.io/)
- [AWS Bedrock Forum](https://forums.aws.amazon.com/)
