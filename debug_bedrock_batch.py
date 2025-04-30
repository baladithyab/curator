#!/usr/bin/env python
"""Debug script for the AWS Bedrock batch processor."""

import os
import tempfile
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor

# Set environment variables
os.environ["CURATOR_LOG_LEVEL"] = "DEBUG"
os.environ["CURATOR_DISABLE_CACHE"] = "1"

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-west-2"

def main():
    """Test the BedrockBatchRequestProcessor."""
    # Create a working directory
    working_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {working_dir}")

    # Create a configuration with a model that supports batch inference
    config = BatchRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 3.5 Sonnet v2 supports batch inference
        generation_params={"temperature": 0.7, "max_tokens": 300}
    )

    # Create the batch processor with actual AWS resources
    try:
        processor = BedrockBatchRequestProcessor(
            config=config,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        )

        # Set the working directory
        processor.working_dir = working_dir

        print(f"Created processor for model: {processor.model_id}")
        print(f"Working directory: {working_dir}")
        print(f"Model provider: {processor.model_provider}")
        print(f"Has prompt_formatter: {hasattr(processor, 'prompt_formatter')}")
        if hasattr(processor, 'prompt_formatter'):
            print(f"Prompt formatter model: {processor.prompt_formatter.model_name}")
        print(f"Has total_requests: {hasattr(processor, 'total_requests')}")
        if hasattr(processor, 'total_requests'):
            print(f"Total requests: {processor.total_requests}")

        return 0
    except Exception as e:
        print(f"Error creating processor: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
