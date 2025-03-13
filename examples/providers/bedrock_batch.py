"""Example of using AWS Bedrock with Curator for batch inference.

This example demonstrates how to use AWS Bedrock for batch processing of requests.
Before running this example, make sure you have the required AWS credentials and permissions.

Required environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-east-1)
- BEDROCK_BATCH_S3_BUCKET (S3 bucket for batch input/output)
- BEDROCK_BATCH_ROLE_ARN (IAM role ARN with necessary permissions)
"""

import os
import time
from bespokelabs import curator

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-east-1"

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
        print("\nPlease set these variables before running the example.")
        return False
    
    return True

def run_claude_batch_example():
    """Run a batch example using Claude from AWS Bedrock."""
    print("\n=== Running Claude Batch Example on AWS Bedrock ===")
    
    # Create some sample requests
    requests = []
    for i in range(5):
        request = curator.GenericRequest(
            task_id=i,
            prompt=f"Write a short paragraph about topic {i+1}: " + [
                "Artificial Intelligence",
                "Cloud Computing",
                "Quantum Computing",
                "Cybersecurity",
                "Machine Learning"
            ][i],
            generation_params={
                "temperature": 0.7,
                "max_tokens": 300
            }
        )
        requests.append(request)
    
    # Create a temporary directory for batch files
    import tempfile
    working_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {working_dir}")
    
    # Create request files
    request_files = []
    for i, batch_requests in enumerate([requests]):  # Single batch for simplicity
        file_path = os.path.join(working_dir, f"batch_{i}.jsonl")
        with open(file_path, "w") as f:
            for request in batch_requests:
                f.write(curator.json.dumps(request.to_dict()) + "\n")
        request_files.append(file_path)
    
    # Get the batch processor
    processor = curator.get_request_processor(
        model_name="anthropic.claude-3-haiku-20240307-v1:0",  # Using smaller model for example
        backend="bedrock",
        batch=True,
        generation_params={"temperature": 0.7, "max_tokens": 300}
    )
    
    # Process the batch
    processor.requests_to_responses(request_files)
    
    # Read and print the results
    print("\nBatch Results:")
    for i, file_path in enumerate(request_files):
        result_path = file_path.replace(".jsonl", "_responses.jsonl")
        if os.path.exists(result_path):
            print(f"\nResults for batch {i}:")
            with open(result_path, "r") as f:
                for j, line in enumerate(f):
                    response = curator.json.loads(line)
                    print(f"\nRequest {j}:")
                    print(f"Prompt: {requests[j].prompt}")
                    print(f"Response: {response.get('response')}")
        else:
            print(f"No results found for batch {i}")

def run_cross_provider_batch_example():
    """Run multiple batch examples with different providers."""
    if not check_environment():
        return
    
    # Define models to test
    models = [
        {"name": "anthropic.claude-3-haiku-20240307-v1:0", "provider": "Claude"},
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
    import tempfile
    working_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {working_dir}")
    
    # Process each model
    for model in models:
        print(f"\n=== Running {model['provider']} Batch Example ===")
        
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
        
        # Get the batch processor
        try:
            processor = curator.get_request_processor(
                model_name=model["name"],
                backend="bedrock",
                batch=True,
                generation_params={"temperature": 0.7, "max_tokens": 300}
            )
            
            # Process the batch
            print(f"Processing batch for {model['provider']}...")
            processor.requests_to_responses([file_path])
            
            # Read and print the results
            result_path = file_path.replace(".jsonl", "_responses.jsonl")
            if os.path.exists(result_path):
                print(f"\nResults for {model['provider']}:")
                with open(result_path, "r") as f:
                    for j, line in enumerate(f):
                        response = curator.json.loads(line)
                        print(f"\nPrompt: {prompts[j]}")
                        print(f"Response: {response.get('response')[:150]}...")  # Print truncated response
            else:
                print(f"No results found for {model['provider']}")
                
        except Exception as e:
            print(f"Error processing {model['provider']}: {str(e)}")
    
    print("\nBatch processing complete.")

# Main function to run the example
def main():
    print("AWS Bedrock Batch Inference Example")
    
    if not check_environment():
        return
    
    try:
        # Run the batch example with Claude
        run_claude_batch_example()
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main() 