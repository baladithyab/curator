"""Example of using AWS Bedrock with Curator for online inference.

This example demonstrates how to use different AWS Bedrock models for text generation.
The BedrockOnlineRequestProcessor automatically uses the Converse API for supported
models (like Claude, Llama, and Titan) to provide better chat capabilities and 
consistent message formatting.

Before running this example, make sure you have the required AWS credentials set up.

Required environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-east-1)
"""

import os
from bespokelabs import curator

import boto3

# Set the AWS region if not already set in environment
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-west-2"

def check_environment():
    """Check if AWS credentials are properly configured."""
    try:
        # Test if we can access AWS credentials (including role-based credentials)
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"Using AWS credentials for: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"Error accessing AWS credentials: {str(e)}")
        print("Make sure you have valid AWS credentials configured.")
        return False

def run_claude_example():
    """Run an example using Claude from AWS Bedrock."""
    print("\n=== Running Claude on AWS Bedrock ===")
    
    llm = curator.LLM(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    messages = [{
        "role": "user",
        "content": [{"text": "Write a limerick about cloud computing."}]
    }]
    response = llm([messages])
    print(f"Response from Claude on AWS Bedrock:\n{response[0]['response']}")

def run_claude_chat_example():
    """Run a chat example using Claude from AWS Bedrock."""
    print("\n=== Running Claude Chat Example on AWS Bedrock ===")
    
    llm = curator.LLM(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    messages = [
        {
            "role": "system",
            "content": [{"text": "You are a helpful AI assistant that specializes in explaining AWS services."}]
        },
        {
            "role": "user",
            "content": [{"text": "Can you explain the benefits of using AWS Bedrock?"}]
        }
    ]
    
    response = llm([messages])
    print(f"Response from Claude Chat:\n{response[0]['response']}")

def run_titan_example():
    """Run an example using Amazon Titan from AWS Bedrock."""
    print("\n=== Running Amazon Titan on AWS Bedrock ===")
    
    llm = curator.LLM(
        model_name="amazon.titan-text-express-v1",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    messages = [{
        "role": "user",
        "content": [{"text": "Write a short poem about machine learning."}]
    }]
    response = llm([messages])
    print(f"Response from Amazon Titan:\n{response[0]['response']}")

def run_llama_example():
    """Run an example using Meta Llama from AWS Bedrock."""
    print("\n=== Running Meta Llama on AWS Bedrock ===")
    
    llm = curator.LLM(
        model_name="meta.llama3-1-8b-instruct-v1:0",  # Use smaller model for example
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )
    
    messages = [{
        "role": "user",
        "content": [{"text": "Explain how transformers work in deep learning."}]
    }]
    response = llm([messages])
    print(f"Response from Meta Llama:\n{response[0]['response']}")

def run_inference_profile_example():
    """Run an example using an inference profile for cross-region availability."""
    print("\n=== Running with Inference Profile ===")
    print("Note: Inference profiles require appropriate AWS permissions and configuration.")
    
    llm = curator.LLM(
        model_name="anthropic.claude-3-haiku-20240307-v1:0",  # Will be converted to profile
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500},
        backend_params={"use_inference_profile": True}  # This enables auto-conversion
    )
    
    messages = [{
        "role": "user",
        "content": [{"text": "Explain the concept of geo-distributed inference and its benefits."}]
    }]
    response = llm([messages])
    print(f"Response with inference profile:\n{response[0]['response']}")

def main():
    """Run the AWS Bedrock online inference examples."""
    print("AWS Bedrock Online Inference Examples")
    
    if not check_environment():
        return
    
    try:
        # Run the examples
        run_claude_example()
        run_claude_chat_example()
        run_titan_example()
        run_llama_example()
        run_inference_profile_example()
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main()
