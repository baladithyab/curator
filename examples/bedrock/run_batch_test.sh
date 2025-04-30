#!/bin/bash
# Script to run the Bedrock batch test with required environment variables

# Set your AWS region
export AWS_REGION=us-west-2

# Set your S3 bucket for batch input/output
# Replace with your actual S3 bucket name
export BEDROCK_BATCH_S3_BUCKET="your-s3-bucket-name"

# Set your IAM role ARN with necessary permissions
# Replace with your actual IAM role ARN
export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/YourBedrockBatchRole"

# Set to 1 for verbose output
export BEDROCK_TEST_VERBOSE=1

# Run the test
python examples/bedrock/test_bedrock_batch_simple.py
