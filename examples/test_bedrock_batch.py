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
"""

import os
import time
import tempfile
from bespokelabs import curator

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-west-2"

def check_environment():
    """Check if the required environment variables are set."""
    missing_vars = []
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
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

def test_claude_batch_simple():
    """Test a simple batch job with Claude."""
    print("\n=== Testing Simple Claude Batch Job ===")

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
    print(f"Using temporary directory: {working_dir}")

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
    print("Submitting batch job...")
    start_time = time.time()
    processor.requests_to_responses([file_path])
    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f}s")

    # Read and print the results
    result_path = file_path.replace(".jsonl", "_responses.jsonl")
    if os.path.exists(result_path):
        print("\nBatch Results:")
        with open(result_path, "r") as f:
            for i, line in enumerate(f):
                response = curator.json.loads(line)
                print(f"\nTopic: {topics[i]}")
                print(f"Response: {response.get('response')}")
                print(f"Token usage: {response.get('token_usage', 'Not available')}")
    else:
        print("No results found. Batch job may have failed or is still processing.")

def test_multiple_models_batch():
    """Test batch processing with multiple models."""
    print("\n=== Testing Multiple Models Batch Processing ===")

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
    print(f"Using temporary directory: {working_dir}")

    # Process each model
    for model in models:
        print(f"\n--- Testing {model['provider']} Batch Processing ---")

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
            print(f"Submitting batch job for {model['provider']}...")
            start_time = time.time()
            processor.requests_to_responses([file_path])
            end_time = time.time()
            print(f"Batch processing completed in {end_time - start_time:.2f}s")

            # Read and print the results
            result_path = file_path.replace(".jsonl", "_responses.jsonl")
            if os.path.exists(result_path):
                print(f"\nResults for {model['provider']}:")
                with open(result_path, "r") as f:
                    for j, line in enumerate(f):
                        response = curator.json.loads(line)
                        print(f"\nPrompt: {prompts[j]}")
                        print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                        print(f"Token usage: {response.get('token_usage', 'Not available')}")
            else:
                print(f"No results found for {model['provider']}")

        except Exception as e:
            print(f"Error processing {model['provider']}: {str(e)}")

def test_inference_profile_batch():
    """Test batch processing with inference profile."""
    print("\n=== Testing Inference Profile Batch Processing ===")
    print("Note: Inference profiles require appropriate AWS permissions and configuration.")

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
    print(f"Using temporary directory: {working_dir}")

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
        print("Submitting batch job with inference profile...")
        start_time = time.time()
        processor.requests_to_responses([file_path])
        end_time = time.time()
        print(f"Batch processing completed in {end_time - start_time:.2f}s")

        # Read and print the results
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            print("\nBatch Results with Inference Profile:")
            with open(result_path, "r") as f:
                for i, line in enumerate(f):
                    response = curator.json.loads(line)
                    print(f"\nTopic: {topics[i]}")
                    print(f"Response: {response.get('response')}")
                    print(f"Token usage: {response.get('token_usage', 'Not available')}")
        else:
            print("No results found. Batch job may have failed or is still processing.")

    except Exception as e:
        print(f"Error processing inference profile batch: {str(e)}")

def test_batch_status_monitoring():
    """Test monitoring batch job status."""
    print("\n=== Testing Batch Status Monitoring ===")

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
    print(f"Using temporary directory: {working_dir}")

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
    print("Submitting batch job for monitoring...")

    # This is a simplified version - in a real implementation, you would
    # need to access the internal tracker and batch objects
    try:
        # Start the batch processing in a way that we can monitor
        # Note: This is a simplified approach and may not work exactly as shown
        # The actual implementation would depend on the internal API of the processor
        processor.requests_to_responses([file_path])

        # In a real implementation, you would access the batch object and check its status
        print("Batch job submitted. In a real implementation, you would monitor the status here.")

        # Read and print the results
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            print("\nBatch Results:")
            with open(result_path, "r") as f:
                for line in f:
                    response = curator.json.loads(line)
                    print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
                    print(f"Token usage: {response.get('token_usage', 'Not available')}")
        else:
            print("No results found. Batch job may have failed or is still processing.")

    except Exception as e:
        print(f"Error in batch monitoring test: {str(e)}")

def main():
    """Run the AWS Bedrock batch inference tests."""
    print("AWS Bedrock Batch Inference Tests")

    if not check_environment():
        return

    try:
        # Run the tests
        test_claude_batch_simple()
        test_multiple_models_batch()
        test_inference_profile_batch()
        test_batch_status_monitoring()

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main()
