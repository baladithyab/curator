# AWS Bedrock Tests

This directory contains test scripts for the AWS Bedrock integration in the Curator project.

## Test Scripts

- `test_bedrock_batch_simple.py`: Simple tests for the Bedrock batch processor that don't require AWS credentials
- `test_bedrock_batch.py`: Tests for the Bedrock batch processor
- `test_bedrock_batch_processor.py`: More comprehensive tests for the Bedrock batch processor
- `test_bedrock_implementation.py`: Tests for the Bedrock implementation
- `test_bedrock_online.py`: Tests for the Bedrock online processor

## Running the Tests

### Prerequisites

To run the tests that make actual API calls, you need:

1. AWS credentials with access to Bedrock configured via AWS CLI
2. Permissions to create S3 buckets and IAM roles (if not providing existing ones)

The test runner will automatically:
- Use the AWS region from your AWS CLI configuration
- Create an S3 bucket and IAM role if they don't exist
- Clean up resources after the tests (unless you specify otherwise)

### Running All Tests

To run all tests with automatic resource creation and cleanup:

```bash
python run_all_tests.py
```

### Command Line Options

The test runner supports several command line options:

```bash
# Run only simple tests that don't require AWS credentials
python run_all_tests.py --simple-only

# Enable verbose output (shows detailed test results)
python run_all_tests.py --verbose

# Keep AWS resources after tests
python run_all_tests.py --keep-resources

# Download test results before cleanup
python run_all_tests.py --download-results

# Specify a download directory
python run_all_tests.py --download-results --download-dir my_results

# Specify an existing S3 bucket to use
python run_all_tests.py --bucket-name my-existing-bucket

# Specify an existing IAM role name to use
python run_all_tests.py --role-name MyExistingRole

# Specify an AWS profile to use
python run_all_tests.py --profile my-profile

# Specify an AWS region to use
python run_all_tests.py --region us-west-2

# Combine options as needed
python run_all_tests.py --simple-only --verbose
```

The test runner will automatically detect and use AWS credentials in the following order of priority:
1. AWS CLI configuration (default profile or specified profile)
2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
3. EC2 instance profile (if running on EC2)

### Setting Up Environment Variables Manually (Optional)

If you prefer to set up environment variables manually:

```bash
# Edit the setup_env.sh script to set your AWS credentials and S3 bucket
nano setup_env.sh

# Source the environment variables
source setup_env.sh
```

### Running Individual Tests

To run a specific test:

```bash
python test_bedrock_batch_simple.py
```

## Test Configuration

The tests use the following environment variables:

- `AWS_REGION`: AWS region to use for Bedrock API calls
- `BEDROCK_BATCH_S3_BUCKET`: S3 bucket to use for batch input/output
- `BEDROCK_BATCH_ROLE_ARN`: IAM role ARN with permissions for Bedrock batch jobs
- `CURATOR_DISABLE_CACHE`: Set to "1" to disable caching for tests
- `CURATOR_LOG_LEVEL`: Set to "DEBUG" for more verbose logging output
- `BEDROCK_TEST_VERBOSE`: Set to "1" to enable verbose test output (same as using --verbose flag)

## Models Tested

The tests cover the following AWS Bedrock models:

- Claude 3.5 Sonnet
- Amazon Nova
- Meta Llama 3
