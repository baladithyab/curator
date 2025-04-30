#!/usr/bin/env python
"""Test script for the AWS Bedrock batch processor.

This script demonstrates how to use the AWS Bedrock batch processor
to process a batch of requests.

Requirements:
- AWS credentials with access to Bedrock
- S3 bucket with write permissions
- IAM role with permissions for Bedrock batch jobs
- BEDROCK_TEST_VERBOSE (optional, set to 1 for verbose output)
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Add the parent directory to the path so we can import test_utils
sys.path.insert(0, str(Path(__file__).parent))
from test_utils import (
    is_verbose, vprint, print_header, print_subheader,
    format_response_summary, print_test_result,
    time_execution, check_aws_environment
)

from bespokelabs import curator
from bespokelabs.curator.types.generic_request import GenericRequest

# Set environment variables
os.environ["CURATOR_DISABLE_CACHE"] = "1"
os.environ["CURATOR_LOG_LEVEL"] = "DEBUG"


def check_environment():
    """Check if the required environment variables are set."""
    required_vars = [
        "BEDROCK_BATCH_S3_BUCKET",
        "BEDROCK_BATCH_ROLE_ARN"
    ]

    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Please set the following environment variables:")
        print("  BEDROCK_BATCH_S3_BUCKET - S3 bucket to use for batch input/output")
        print("  BEDROCK_BATCH_ROLE_ARN - IAM role ARN with permissions for Bedrock batch jobs")
        return False

    return True


@time_execution
def test_bedrock_batch():
    """Test the AWS Bedrock batch processor with Claude 3.5 Sonnet."""
    print_subheader("Testing AWS Bedrock Batch Processor with Claude 3.5 Sonnet")

    # Create sample prompts
    prompts = [
        "Explain the concept of quantum computing in simple terms.",
        "What are the key differences between machine learning and deep learning?",
        "How does natural language processing work?",
        "Explain the concept of blockchain technology.",
        "What are the ethical considerations in artificial intelligence?"
    ]

    # Create generic requests
    requests = []
    for i, prompt in enumerate(prompts):
        request = GenericRequest(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt=prompt,
            generation_params={"temperature": 0.7, "max_tokens": 300},
            original_row={"prompt": prompt},
            original_row_idx=i,
            task_id=i
        )
        requests.append(request)

    # Create a temporary directory for batch files
    working_dir = tempfile.mkdtemp()
    vprint(f"Using temporary directory: {working_dir}")

    # Print a summary in non-verbose mode
    if not is_verbose():
        print(f"Created temporary working directory for batch files")

    # Create request file
    file_path = os.path.join(working_dir, "claude_batch.jsonl")
    with open(file_path, "w") as f:
        for request in requests:
            f.write(curator.json.dumps(request.to_dict()) + "\n")

    # Get the batch processor
    processor = curator.get_request_processor(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        backend="bedrock",
        batch=True,
        generation_params={"temperature": 0.7, "max_tokens": 300},
        backend_params={
            "s3_bucket": os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            "s3_prefix": "curator-test",
            "role_arn": os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        }
    )

    # Process the batch
    vprint("Processing batch...")
    if not is_verbose():
        print("Processing batch with Claude 3.5 Sonnet...")

    start_time = time.time()
    processor.requests_to_responses([file_path])
    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f}s")

    # Read and print the results
    result_path = file_path.replace("claude_batch.jsonl", "responses_claude_batch.jsonl")
    if os.path.exists(result_path):
        vprint("\nResults:")
        with open(result_path, "r") as f:
            for j, line in enumerate(f):
                response = curator.json.loads(line)
                if is_verbose():
                    print(f"\nPrompt: {prompts[j]}")
                    print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                else:
                    # In non-verbose mode, just print a short summary for the first response
                    if j == 0:
                        summary = format_response_summary(prompts[j], response)
                        print(f"Sample response: {summary}")
        return True
    else:
        print("No results found. Check if the batch job completed successfully.")
        return False


@time_execution
def test_multiple_models_batch():
    """Test batch processing with multiple models."""
    print_subheader("Testing Multiple Models Batch Processing")

    # Define models to test
    models = [
        {"name": "anthropic.claude-3-5-sonnet-20240620-v1:0", "provider": "Claude"},
        {"name": "amazon.nova-lite-v1:0", "provider": "Amazon Nova"},
        {"name": "meta.llama3-3-70b-instruct-v1:0", "provider": "Meta Llama"}
    ]

    # Create sample prompts
    prompts = [
        "Explain the concept of microservices in software architecture.",
        "What are the key benefits of serverless computing?",
        "How does containerization improve application deployment?"
    ]

    # Create a temporary directory for batch files
    working_dir = tempfile.mkdtemp()
    vprint(f"Using temporary directory: {working_dir}")

    # Print a summary in non-verbose mode
    if not is_verbose():
        print(f"Created temporary working directory for multiple model batch files")

    # Track success for each model
    all_models_success = True

    # Process each model
    for model in models:
        vprint(f"\nTesting {model['provider']} ({model['name']})...")
        if not is_verbose():
            print(f"Testing {model['provider']} batch processing...")

        # Create generic requests
        requests = []
        for i, prompt in enumerate(prompts):
            request = GenericRequest(
                model=model["name"],
                prompt=prompt,
                generation_params={"temperature": 0.7, "max_tokens": 300},
                original_row={"prompt": prompt},
                original_row_idx=i,
                task_id=i
            )
            requests.append(request)

        # Create request file
        file_path = os.path.join(working_dir, f"{model['provider'].lower()}_batch.jsonl")
        with open(file_path, "w") as f:
            for request in requests:
                f.write(curator.json.dumps(request.to_dict()) + "\n")

        # Get the batch processor
        try:
            processor = curator.get_request_processor(
                model_name=model["name"],
                backend="bedrock",
                batch=True,
                generation_params={"temperature": 0.7, "max_tokens": 300},
                backend_params={
                    "s3_bucket": os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
                    "s3_prefix": "curator-test",
                    "role_arn": os.environ.get("BEDROCK_BATCH_ROLE_ARN")
                }
            )

            # Process the batch
            vprint(f"Processing batch for {model['provider']}...")
            start_time = time.time()
            processor.requests_to_responses([file_path])
            end_time = time.time()
            print(f"{model['provider']} batch processing completed in {end_time - start_time:.2f}s")

            # Read and print the results
            result_path = file_path.replace(".jsonl", "_responses.jsonl")
            if os.path.exists(result_path):
                vprint(f"\nResults for {model['provider']}:")
                with open(result_path, "r") as f:
                    for j, line in enumerate(f):
                        response = curator.json.loads(line)
                        if is_verbose():
                            print(f"\nPrompt: {prompts[j]}")
                            print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                        else:
                            # In non-verbose mode, just print a short summary for the first response
                            if j == 0:
                                summary = format_response_summary(prompts[j], response)
                                print(f"Sample {model['provider']} response: {summary}")
            else:
                print(f"No results found for {model['provider']}")
                all_models_success = False
        except Exception as e:
            print(f"Error processing {model['provider']}: {str(e)}")
            print("Make sure you have access to this model in AWS Bedrock.")
            all_models_success = False

    return all_models_success


@time_execution
def test_inference_profile_batch():
    """Test batch processing with inference profiles."""
    print_subheader("Testing Inference Profile Batch Processing")

    # Create sample prompts
    prompts = [
        "What are the advantages of using cross-region inference profiles in AWS Bedrock?",
        "How do inference profiles improve model availability and latency?"
    ]

    # Create generic requests
    requests = []
    for i, prompt in enumerate(prompts):
        request = GenericRequest(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Will be converted to inference profile
            prompt=prompt,
            generation_params={"temperature": 0.7, "max_tokens": 300},
            original_row={"prompt": prompt},
            original_row_idx=i,
            task_id=i
        )
        requests.append(request)

    # Create a temporary directory for batch files
    working_dir = tempfile.mkdtemp()
    vprint(f"Using temporary directory: {working_dir}")

    # Print a summary in non-verbose mode
    if not is_verbose():
        print(f"Created temporary working directory for inference profile batch files")

    # Create request file
    file_path = os.path.join(working_dir, "profile_batch.jsonl")
    with open(file_path, "w") as f:
        for request in requests:
            f.write(curator.json.dumps(request.to_dict()) + "\n")

    try:
        # Get the batch processor with inference profile
        processor = curator.get_request_processor(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            backend="bedrock",
            batch=True,
            generation_params={"temperature": 0.7, "max_tokens": 300},
            backend_params={
                "s3_bucket": os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
                "s3_prefix": "curator-test-profile",
                "role_arn": os.environ.get("BEDROCK_BATCH_ROLE_ARN"),
                "use_inference_profile": True
            }
        )

        # Process the batch
        vprint("Processing batch with inference profile...")
        if not is_verbose():
            print("Processing batch with inference profile...")

        start_time = time.time()
        processor.requests_to_responses([file_path])
        end_time = time.time()
        print(f"Inference profile batch processing completed in {end_time - start_time:.2f}s")

        # Read and print the results
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            vprint("\nResults with inference profile:")
            with open(result_path, "r") as f:
                for j, line in enumerate(f):
                    response = curator.json.loads(line)
                    if is_verbose():
                        print(f"\nPrompt: {prompts[j]}")
                        print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                    else:
                        # In non-verbose mode, just print a short summary for the first response
                        if j == 0:
                            summary = format_response_summary(prompts[j], response)
                            print(f"Sample inference profile response: {summary}")
            return True
        else:
            print("No results found for inference profile batch")
            return False
    except Exception as e:
        print(f"Error processing inference profile batch: {str(e)}")
        return False


def main():
    """Run the AWS Bedrock batch inference tests."""
    print_header("AWS BEDROCK BATCH PROCESSOR TESTS")

    # Check AWS environment
    aws_ok, _ = check_aws_environment()
    if not aws_ok:
        return 1

    # Check if batch environment variables are set
    if not check_environment():
        return 1

    try:
        # Run the tests and collect results
        results = {}
        results["Claude Batch"] = test_bedrock_batch()
        results["Multiple Models Batch"] = test_multiple_models_batch()
        results["Inference Profile Batch"] = test_inference_profile_batch()

        # Print summary
        print_subheader("TEST SUMMARY")
        all_passed = True
        for test_name, passed in results.items():
            print_test_result(test_name, passed)
            if not passed:
                all_passed = False

        return 0 if all_passed else 1

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
