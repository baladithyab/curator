"""Example of using AWS Bedrock with Curator for online inference.

This example demonstrates how to use different AWS Bedrock models for text generation.
Before running this example, make sure you have the required AWS credentials set up.

Required environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-east-1)
"""

import os
from bespokelabs import curator

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-east-1"

# Get the request processor for AWS Bedrock with Claude model
def run_claude_example():
    """Run an example using Claude from AWS Bedrock."""
    print("\n=== Running Claude on AWS Bedrock ===")
    # Get the processor with explicit 'bedrock' backend
    processor = curator.get_request_processor(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    response = processor.generate("Write a limerick about cloud computing.")
    print(f"Response from Claude on AWS Bedrock:\n{response}")

# Get the request processor for AWS Bedrock with Titan model
def run_titan_example():
    """Run an example using Amazon Titan from AWS Bedrock."""
    print("\n=== Running Amazon Titan on AWS Bedrock ===")
    processor = curator.get_request_processor(
        model_name="amazon.titan-text-express-v1",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    response = processor.generate("Write a short poem about machine learning.")
    print(f"Response from Amazon Titan on AWS Bedrock:\n{response}")

# Get the request processor for AWS Bedrock with Llama model
def run_llama_example():
    """Run an example using Meta Llama from AWS Bedrock."""
    print("\n=== Running Meta Llama on AWS Bedrock ===")
    processor = curator.get_request_processor(
        model_name="meta.llama3-1-8b-instruct-v1:0",  # Use smaller model for example
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    response = processor.generate("Explain how transformers work in deep learning.")
    print(f"Response from Meta Llama on AWS Bedrock:\n{response}")

# Run with auto-detected model (using model name prefix)
def run_auto_detected_example():
    """Run an example with auto-detected backend based on model ID."""
    print("\n=== Running with Auto-detected Backend ===")
    # The factory will automatically detect this as a Bedrock model
    processor = curator.get_request_processor(
        model_name="anthropic.claude-3-haiku-20240307-v1:0",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    response = processor.generate("What are the benefits of serverless architecture?")
    print(f"Response with auto-detected backend:\n{response}")

# Main function to run all examples
def main():
    print("AWS Bedrock Online Inference Examples")
    
    # Check for AWS credentials
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("Warning: AWS credentials not found in environment variables.")
        print("Make sure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
    
    # Run the examples
    try:
        run_claude_example()
        run_titan_example()
        run_llama_example()
        run_auto_detected_example()
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main() 