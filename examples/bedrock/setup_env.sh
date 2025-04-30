#!/bin/bash
# Setup environment variables for Bedrock tests

# AWS Region
export AWS_REGION="us-east-1"

# S3 bucket for batch processing
# Replace with your actual S3 bucket name
export BEDROCK_BATCH_S3_BUCKET="your-s3-bucket-name"

# IAM role ARN for Bedrock batch processing
# Replace with your actual IAM role ARN
export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/YourBedrockBatchRole"

# Disable caching for tests
export CURATOR_DISABLE_CACHE="1"

# Set log level to DEBUG for more verbose output
export CURATOR_LOG_LEVEL="DEBUG"

echo "Environment variables set for Bedrock tests:"
echo "AWS_REGION: $AWS_REGION"
echo "BEDROCK_BATCH_S3_BUCKET: $BEDROCK_BATCH_S3_BUCKET"
echo "BEDROCK_BATCH_ROLE_ARN: $BEDROCK_BATCH_ROLE_ARN"
echo "CURATOR_DISABLE_CACHE: $CURATOR_DISABLE_CACHE"
echo "CURATOR_LOG_LEVEL: $CURATOR_LOG_LEVEL"
