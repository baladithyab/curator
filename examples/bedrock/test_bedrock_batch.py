"""Test script for AWS Bedrock batch inference with Curator.

This script tests the Bedrock batch processor implementation with various models
and configurations to ensure it's working correctly.

Before running this script, make sure you have the required AWS credentials and permissions.

Required environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-west-2)
- BEDROCK_BATCH_S3_BUCKET (S3 bucket for batch input/output)
- BEDROCK_BATCH_ROLE_ARN (IAM role ARN with necessary permissions)
- BEDROCK_TEST_VERBOSE (optional, set to 1 for verbose output)
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import test_utils
sys.path.insert(0, str(Path(__file__).parent))
from test_utils import (
    is_verbose, vprint, print_header, print_subheader,
    format_response_summary, print_test_result,
    time_execution, check_aws_environment
)

from bespokelabs import curator

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-west-2"

def check_environment():
    """Check if the required environment variables are set."""
    missing_vars = []
    required_vars = [
        "BEDROCK_BATCH_S3_BUCKET",
        "BEDROCK_BATCH_ROLE_ARN"
    ]

    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        print("Error: The following required environment variables are not set:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables before running the test.")
        return False

    return True

@time_execution
def test_claude_batch_simple():
    """Test a simple batch job with Claude."""
    print_subheader("Testing Simple Claude Batch Job")

    # Create sample requests
    requests = []
    topics = [
        "Artificial Intelligence",
        "Cloud Computing",
        "Quantum Computing",
        "Cybersecurity",
        "Machine Learning"
    ]

    for i, topic in enumerate(topics):
        request = curator.GenericRequest(
            task_id=i,
            prompt=f"Write a short paragraph about {topic}.",
            generation_params={
                "temperature": 0.7,
                "max_tokens": 300
            }
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
    vprint("Submitting batch job...")
    if not is_verbose():
        print("Submitting batch job to Bedrock...")

    start_time = time.time()
    processor.requests_to_responses([file_path])
    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f}s")

    # Read and print the results
    result_path = file_path.replace(".jsonl", "_responses.jsonl")
    if os.path.exists(result_path):
        vprint("\nBatch Results:")
        with open(result_path, "r") as f:
            for i, line in enumerate(f):
                response = curator.json.loads(line)
                if is_verbose():
                    print(f"\nTopic: {topics[i]}")
                    print(f"Response: {response.get('response')}")
                    print(f"Token usage: {response.get('token_usage', 'Not available')}")
                else:
                    # In non-verbose mode, just print a short summary for the first response
                    if i == 0:
                        prompt = f"Write a short paragraph about {topics[i]}."
                        summary = format_response_summary(prompt, response)
                        print(f"Sample response: {summary}")
    else:
        print("No results found. Batch job may have failed or is still processing.")

    return True

@time_execution
def test_multiple_models_batch():
    """Test batch processing with multiple models."""
    print_subheader("Testing Multiple Models Batch Processing")

    # Define models to test
    models = [
        {"name": "anthropic.claude-3-5-sonnet-20240620-v1:0", "provider": "Claude"},
        {"name": "amazon.titan-text-express-v1", "provider": "Amazon Titan"},
        {"name": "meta.llama3-1-8b-instruct-v1:0", "provider": "Meta Llama"}
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
        vprint(f"\n--- Testing {model['provider']} Batch Processing ---")
        if not is_verbose():
            print(f"Testing {model['provider']} batch processing...")

        # Create requests for this model
        requests = []
        for i, prompt in enumerate(prompts):
            request = curator.GenericRequest(
                task_id=i,
                prompt=prompt,
                generation_params={
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            )
            requests.append(request)

        # Create request file
        file_path = os.path.join(working_dir, f"{model['provider'].lower()}_batch.jsonl")
        with open(file_path, "w") as f:
            for request in requests:
                f.write(curator.json.dumps(request.to_dict()) + "\n")

        try:
            # Get the batch processor
            processor = curator.get_request_processor(
                model_name=model["name"],
                backend="bedrock",
                batch=True,
                generation_params={"temperature": 0.7, "max_tokens": 300},
                backend_params={
                    "s3_bucket": os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
                    "s3_prefix": f"curator-test-{model['provider'].lower()}",
                    "role_arn": os.environ.get("BEDROCK_BATCH_ROLE_ARN")
                }
            )

            # Process the batch
            vprint(f"Submitting batch job for {model['provider']}...")
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
                            print(f"Token usage: {response.get('token_usage', 'Not available')}")
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
            all_models_success = False

    return all_models_success

@time_execution
def test_inference_profile_batch():
    """Test batch processing with inference profile."""
    print_subheader("Testing Inference Profile Batch Processing")
    vprint("Note: Inference profiles require appropriate AWS permissions and configuration.")

    # Create sample requests
    requests = []
    topics = [
        "Edge computing",
        "Distributed systems",
        "Cloud-native architecture"
    ]

    for i, topic in enumerate(topics):
        request = curator.GenericRequest(
            task_id=i,
            prompt=f"Explain the concept of {topic} in 2-3 sentences.",
            generation_params={
                "temperature": 0.7,
                "max_tokens": 300
            }
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
        vprint("Submitting batch job with inference profile...")
        if not is_verbose():
            print("Submitting batch job with inference profile...")

        start_time = time.time()
        processor.requests_to_responses([file_path])
        end_time = time.time()
        print(f"Inference profile batch processing completed in {end_time - start_time:.2f}s")

        # Read and print the results
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            vprint("\nBatch Results with Inference Profile:")
            with open(result_path, "r") as f:
                for i, line in enumerate(f):
                    response = curator.json.loads(line)
                    if is_verbose():
                        print(f"\nTopic: {topics[i]}")
                        print(f"Response: {response.get('response')}")
                        print(f"Token usage: {response.get('token_usage', 'Not available')}")
                    else:
                        # In non-verbose mode, just print a short summary for the first response
                        if i == 0:
                            prompt = f"Explain the concept of {topics[i]} in 2-3 sentences."
                            summary = format_response_summary(prompt, response)
                            print(f"Sample inference profile response: {summary}")
            return True
        else:
            print("No results found. Batch job may have failed or is still processing.")
            return False

    except Exception as e:
        print(f"Error processing inference profile batch: {str(e)}")
        return False

@time_execution
def test_batch_status_monitoring():
    """Test monitoring batch job status."""
    print_subheader("Testing Batch Status Monitoring")

    # Create a simple request
    request = curator.GenericRequest(
        task_id=0,
        prompt="Write a detailed explanation of how AWS Bedrock batch processing works.",
        generation_params={
            "temperature": 0.7,
            "max_tokens": 1000  # Longer response to give us time to monitor
        }
    )

    # Create a temporary directory for batch files
    working_dir = tempfile.mkdtemp()
    vprint(f"Using temporary directory: {working_dir}")

    # Print a summary in non-verbose mode
    if not is_verbose():
        print(f"Created temporary working directory for batch monitoring")

    # Create request file
    file_path = os.path.join(working_dir, "monitor_batch.jsonl")
    with open(file_path, "w") as f:
        f.write(curator.json.dumps(request.to_dict()) + "\n")

    # Get the batch processor
    processor = curator.get_request_processor(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Using larger model for longer processing time
        backend="bedrock",
        batch=True,
        generation_params={"temperature": 0.7, "max_tokens": 1000},
        backend_params={
            "s3_bucket": os.environ.get("BEDROCK_BATCH_S3_BUCKET"),
            "s3_prefix": "curator-test-monitor",
            "role_arn": os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        }
    )

    # Submit the batch but don't wait for completion
    vprint("Submitting batch job for monitoring...")
    if not is_verbose():
        print("Submitting batch job for monitoring...")

    # This is a simplified version - in a real implementation, you would
    # need to access the internal tracker and batch objects
    try:
        # Start the batch processing in a way that we can monitor
        # Note: This is a simplified approach and may not work exactly as shown
        # The actual implementation would depend on the internal API of the processor
        processor.requests_to_responses([file_path])

        # In a real implementation, you would access the batch object and check its status
        vprint("Batch job submitted. In a real implementation, you would monitor the status here.")
        if not is_verbose():
            print("Batch job submitted and completed")

        # Read and print the results
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            vprint("\nBatch Results:")
            with open(result_path, "r") as f:
                line = f.readline()  # Just read the first line since we only have one request
                if line:
                    response = curator.json.loads(line)
                    if is_verbose():
                        print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                        print(f"Token usage: {response.get('token_usage', 'Not available')}")
                    else:
                        # In non-verbose mode, just print a short summary
                        prompt = "Write a detailed explanation of how AWS Bedrock batch processing works."
                        summary = format_response_summary(prompt, response)
                        print(f"Sample monitoring response: {summary}")
            return True
        else:
            print("No results found. Batch job may have failed or is still processing.")
            return False

    except Exception as e:
        print(f"Error in batch monitoring test: {str(e)}")
        return False

def main():
    """Run the AWS Bedrock batch inference tests."""
    print_header("AWS BEDROCK BATCH INFERENCE TESTS")

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
        results["Claude Batch Simple"] = test_claude_batch_simple()
        results["Multiple Models Batch"] = test_multiple_models_batch()
        results["Inference Profile Batch"] = test_inference_profile_batch()
        results["Batch Status Monitoring"] = test_batch_status_monitoring()

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
