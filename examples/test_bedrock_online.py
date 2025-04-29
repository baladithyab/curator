"""Test script for AWS Bedrock online inference with Curator.

This script tests the Bedrock online processor implementation with various models
and configurations to ensure it's working correctly.

Before running this script, make sure you have the required AWS credentials set up.

Required environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional, defaults to us-west-2)
"""

import os
import time

# Set environment variable to disable caching
os.environ["CURATOR_DISABLE_CACHE"] = "1"
from bespokelabs import curator

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

def test_claude_simple_prompt():
    """Test Claude with a simple prompt."""
    print("\n=== Testing Claude with Simple Prompt ===")

    llm = curator.LLM(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Create 5 prompts for batch processing
    prompts = [
        "Write a short poem about artificial intelligence.",
        "Write a short poem about machine learning.",
        "Write a short poem about neural networks.",
        "Write a short poem about deep learning.",
        "Write a short poem about natural language processing."
    ]
    print(f"Processing {len(prompts)} prompts in a batch")

    start_time = time.time()
    responses = llm(prompts)
    end_time = time.time()

    print(f"Responses from Claude (took {end_time - start_time:.2f}s):")
    try:
        for i, response in enumerate(responses):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            if isinstance(response, dict):
                print(response["response"])
                print(f"Token usage: {response.get('token_usage', 'Not available')}")
            elif hasattr(response, 'response'):
                print(response.response)
                print(f"Token usage: {getattr(response, 'token_usage', 'Not available')}")
            else:
                print(str(response))
                print("Token usage: Not available")
    except Exception as e:
        print(f"Error accessing response: {str(e)}")
        print(f"Response type: {type(responses)}")
        print(f"Response content: {str(responses)}")

def test_claude_chat_messages():
    """Test Claude with chat messages."""
    print("\n=== Testing Claude with Chat Messages ===")

    llm = curator.LLM(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Create 5 different chat message sets
    topics = [
        "quantum computing",
        "black holes",
        "DNA",
        "climate change",
        "artificial intelligence"
    ]

    all_messages = []
    for topic in topics:
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful AI assistant that specializes in explaining complex topics simply."}]
            },
            {
                "role": "user",
                "content": [{"text": f"Explain {topic} to a 10-year-old."}]
            }
        ]
        all_messages.append(messages)

    print(f"Processing {len(all_messages)} chat message sets in a batch")

    start_time = time.time()
    responses = llm(all_messages)
    end_time = time.time()

    print(f"Responses from Claude Chat (took {end_time - start_time:.2f}s):")
    for i, response in enumerate(responses):
        print(f"\nTopic {i+1}: {topics[i]}")
        if isinstance(response, dict):
            print(response["response"])
            print(f"Token usage: {response.get('token_usage', 'Not available')}")
        else:
            print(response.response)
            print(f"Token usage: {getattr(response, 'token_usage', 'Not available')}")

def test_nova_model():
    """Test Amazon Nova model."""
    print("\n=== Testing Amazon Nova Model ===")

    llm = curator.LLM(
        model_name="us.amazon.nova-pro-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Create 5 prompts for batch processing
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Explain the concept of neural networks in simple terms.",
        "Explain the concept of deep learning in simple terms.",
        "Explain the concept of natural language processing in simple terms.",
        "Explain the concept of computer vision in simple terms."
    ]
    print(f"Processing {len(prompts)} prompts in a batch")

    start_time = time.time()
    responses = llm(prompts)
    end_time = time.time()

    print(f"Responses from Nova (took {end_time - start_time:.2f}s):")
    for i, response in enumerate(responses):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        if isinstance(response, dict):
            print(response["response"])
            print(f"Token usage: {response.get('token_usage', 'Not available')}")
        elif hasattr(response, 'response'):
            print(response.response)
            print(f"Token usage: {getattr(response, 'token_usage', 'Not available')}")
        else:
            print(str(response))
            print("Token usage: Not available")

def test_llama_model():
    """Test Meta Llama model."""
    print("\n=== Testing Meta Llama Model ===")

    # Use a different Llama model that doesn't require an inference profile
    llm = curator.LLM(
        model_name="meta.llama3-8b-instruct-v1:0",
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Create 5 prompts for batch processing
    prompts = [
        "What are the key differences between supervised and unsupervised learning?",
        "What are the key differences between classification and regression?",
        "What are the key differences between deep learning and machine learning?",
        "What are the key differences between CNN and RNN?",
        "What are the key differences between overfitting and underfitting?"
    ]
    print(f"Processing {len(prompts)} prompts in a batch")

    start_time = time.time()
    responses = llm(prompts)
    end_time = time.time()

    print(f"Responses from Llama (took {end_time - start_time:.2f}s):")
    for i, response in enumerate(responses):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        if isinstance(response, dict):
            print(response["response"])
            print(f"Token usage: {response.get('token_usage', 'Not available')}")
        elif hasattr(response, 'response'):
            print(response.response)
            print(f"Token usage: {getattr(response, 'token_usage', 'Not available')}")
        else:
            print(str(response))
            print("Token usage: Not available")

def test_inference_profile():
    """Test using an inference profile."""
    print("\n=== Testing Inference Profile ===")
    print("Note: Inference profiles require appropriate AWS permissions and configuration.")

    # Skip this test for now
    print("Skipping inference profile test for now.")
    return



def main():
    """Run the AWS Bedrock online inference tests."""
    print("AWS Bedrock Online Inference Tests")

    if not check_environment():
        return

    try:
        # Run the tests with job size of 5 for each test
        print("\nRunning tests with job size of 5 for each model\n")

        # Run the tests
        test_claude_simple_prompt()
        test_claude_chat_messages()
        test_nova_model()
        test_llama_model()
        test_inference_profile()

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        print("Make sure you have access to the specified models in AWS Bedrock.")

if __name__ == "__main__":
    main()
