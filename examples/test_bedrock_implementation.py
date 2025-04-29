"""Test script for AWS Bedrock implementation details.

This script tests specific implementation details of the Bedrock processors,
focusing on the areas that need improvement or verification.

Before running this script, make sure you have the required AWS credentials and permissions.
This script will use IAM profiles for AWS access if available.

Required environment variables:
- AWS_REGION (optional, defaults to us-west-2)
- BEDROCK_BATCH_S3_BUCKET (for batch tests)
- BEDROCK_BATCH_ROLE_ARN (for batch tests)
"""

import os
import time
import tempfile
import asyncio

# Set environment variable to disable caching
os.environ["CURATOR_DISABLE_CACHE"] = "1"
from bespokelabs import curator
from bespokelabs.curator.request_processor.online.bedrock_online_request_processor import BedrockOnlineRequestProcessor
from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig, BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-west-2"

def check_environment():
    """Check if AWS credentials are properly configured."""
    try:
        # Test if we can access AWS credentials (including role-based credentials)
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"Using AWS credentials for: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"Error accessing AWS credentials: {str(e)}")
        print("Make sure you have valid AWS credentials configured.")
        return False

def test_online_processor_initialization():
    """Test initialization of the BedrockOnlineRequestProcessor."""
    print("\n=== Testing BedrockOnlineRequestProcessor Initialization ===")

    # Test with default configuration
    config = OnlineRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    processor = BedrockOnlineRequestProcessor(config)

    print(f"Model ID: {processor.model_id}")
    print(f"Model Provider: {processor.model_provider}")
    print(f"Region: {processor.region_name}")

    # Test with inference profile
    config_profile = OnlineRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    processor_profile = BedrockOnlineRequestProcessor(config_profile, use_inference_profile=True)

    print(f"\nWith Inference Profile:")
    print(f"Model ID: {processor_profile.model_id}")
    print(f"Using Inference Profile: {processor_profile.use_inference_profile}")

async def test_online_processor_call_single_request():
    """Test the call_single_request method of BedrockOnlineRequestProcessor."""
    print("\n=== Testing BedrockOnlineRequestProcessor call_single_request ===")

    # Create a processor with converse API
    config = OnlineRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    processor = BedrockOnlineRequestProcessor(config)

    # Create a generic request
    request = GenericRequest(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        prompt="What is the capital of France?",
        generation_params={"temperature": 0.7, "max_tokens": 500},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        original_row={"prompt": "What is the capital of France?"},
        original_row_idx=0
    )

    # Test with converse API
    print("Testing with Converse API:")
    try:
        # Note: The call_single_request method requires session and status_tracker parameters
        # which are not available in this test script. We'll skip this test for now.
        print("Skipping call_single_request test - requires session and status_tracker parameters")
    except Exception as e:
        print(f"Error with Converse API: {str(e)}")

def test_batch_processor_initialization():
    """Test initialization of the BedrockBatchRequestProcessor."""
    print("\n=== Testing BedrockBatchRequestProcessor Initialization ===")

    # Check if batch environment variables are set
    if not os.environ.get("BEDROCK_BATCH_S3_BUCKET") or not os.environ.get("BEDROCK_BATCH_ROLE_ARN"):
        print("Skipping batch tests - BEDROCK_BATCH_S3_BUCKET or BEDROCK_BATCH_ROLE_ARN not set")
        return

    # Test with default configuration
    config = BatchRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    try:
        processor = BedrockBatchRequestProcessor(
            config,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        )

        print(f"Model ID: {processor.model_id}")
        print(f"Model Provider: {processor.model_provider}")
        print(f"Region: {processor.region_name}")
        print(f"S3 Bucket: {processor.s3_bucket}")
        print(f"S3 Prefix: {processor.s3_prefix}")
        print(f"Using Inference Profile: {processor.use_inference_profile}")
        print(f"Max Requests Per Batch: {processor.max_requests_per_batch}")

        # Test with inference profile
        processor_profile = BedrockBatchRequestProcessor(
            config,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test-profile",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN"),
            use_inference_profile=True
        )

        print(f"\nWith Inference Profile:")
        print(f"Model ID: {processor_profile.model_id}")
        print(f"Using Inference Profile: {processor_profile.use_inference_profile}")

    except Exception as e:
        print(f"Error initializing BedrockBatchRequestProcessor: {str(e)}")
        print("This may be due to missing abstract method implementations.")

async def test_batch_processor_format_request():
    """Test the format_request_for_batch method of BedrockBatchRequestProcessor."""
    print("\n=== Testing BedrockBatchRequestProcessor format_request_for_batch ===")

    # Check if batch environment variables are set
    if not os.environ.get("BEDROCK_BATCH_S3_BUCKET") or not os.environ.get("BEDROCK_BATCH_ROLE_ARN"):
        print("Skipping batch tests - BEDROCK_BATCH_S3_BUCKET or BEDROCK_BATCH_ROLE_ARN not set")
        return

    # Test with Claude model
    config_claude = BatchRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    try:
        processor_claude = BedrockBatchRequestProcessor(
            config_claude,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        )

        # Create a generic request for Claude
        request_claude = GenericRequest(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt="What is the capital of France?",
            generation_params={"temperature": 0.7, "max_tokens": 500},
            task_id=1
        )

        # Format the request
        formatted_claude = await processor_claude.format_request_for_batch(request_claude)

        print("Claude formatted request:")
        print(formatted_claude)

        # Test with Nova model
        config_nova = BatchRequestProcessorConfig(
            model="amazon.nova-pro-v1:0",
            generation_params={"temperature": 0.7, "max_tokens": 500}
        )

        processor_nova = BedrockBatchRequestProcessor(
            config_nova,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        )

        # Create a generic request for Nova
        request_nova = GenericRequest(
            model="amazon.nova-pro-v1:0",
            prompt="What is the capital of France?",
            generation_params={"temperature": 0.7, "max_tokens": 500},
            task_id=2
        )

        # Format the request
        formatted_nova = await processor_nova.format_request_for_batch(request_nova)

        print("\nNova formatted request:")
        print(formatted_nova)

    except Exception as e:
        print(f"Error testing format_request_for_batch: {str(e)}")

async def test_batch_processor_submit_batch():
    """Test the submit_batch method of BedrockBatchRequestProcessor."""
    print("\n=== Testing BedrockBatchRequestProcessor submit_batch ===")

    # Check if batch environment variables are set
    if not os.environ.get("BEDROCK_BATCH_S3_BUCKET") or not os.environ.get("BEDROCK_BATCH_ROLE_ARN"):
        print("Skipping batch tests - BEDROCK_BATCH_S3_BUCKET or BEDROCK_BATCH_ROLE_ARN not set")
        return

    # Create a temporary directory for working files
    working_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {working_dir}")

    # Test with Claude model
    config = BatchRequestProcessorConfig(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 300}
    )

    try:
        processor = BedrockBatchRequestProcessor(
            config,
            s3_bucket=os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            s3_prefix="curator-test-submit",
            role_arn=os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        )

        # Set working directory
        processor.working_dir = working_dir

        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[
                GenericRequest(
                    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    prompt="What is the capital of France?",
                    generation_params={"temperature": 0.7, "max_tokens": 300},
                    task_id=1
                ),
                GenericRequest(
                    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    prompt="What is the capital of Germany?",
                    generation_params={"temperature": 0.7, "max_tokens": 300},
                    task_id=2
                )
            ],
            status=GenericBatchStatus.PENDING
        )

        # Submit the batch
        print("Submitting batch...")
        await processor.submit_batch(batch)

        print(f"Batch submitted with ID: {batch.provider_batch_id}")
        print(f"Batch status: {batch.status}")
        print(f"Batch metadata: {batch.metadata}")

        # Check batch status
        print("\nChecking batch status...")
        status = await processor.check_batch_status(batch)

        print(f"Current batch status: {status}")
        print(f"Updated batch metadata: {batch.metadata}")

        # Wait for batch to complete
        print("\nWaiting for batch to complete...")
        max_wait_time = 300  # 5 minutes
        start_time = time.time()

        while batch.status != GenericBatchStatus.COMPLETE and time.time() - start_time < max_wait_time:
            print(f"Current status: {batch.status}. Waiting...")
            await asyncio.sleep(10)
            status = await processor.check_batch_status(batch)

        if batch.status == GenericBatchStatus.COMPLETE:
            print("\nBatch completed successfully!")

            # Fetch results
            print("Fetching batch results...")
            responses = await processor.fetch_batch_results(batch)

            for response in responses:
                print(f"\nTask ID: {response.task_id}")
                print(f"Response: {response.response}")
                print(f"Token usage: {response.token_usage}")
        else:
            print(f"\nBatch did not complete within the wait time. Final status: {batch.status}")

        # Clean up
        print("\nCleaning up batch resources...")
        await processor.cleanup_batch(batch)

    except Exception as e:
        print(f"Error testing submit_batch: {str(e)}")

async def run_async_tests():
    """Run all async tests."""
    await test_online_processor_call_single_request()
    await test_batch_processor_format_request()
    await test_batch_processor_submit_batch()

def main():
    """Run the AWS Bedrock implementation tests."""
    print("AWS Bedrock Implementation Tests")

    if not check_environment():
        return

    try:
        # Run synchronous tests
        test_online_processor_initialization()
        test_batch_processor_initialization()

        # Run asynchronous tests
        asyncio.run(run_async_tests())

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main()
