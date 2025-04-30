"""Comprehensive examples demonstrating how to use Curator with AWS Bedrock for online inference.

This example shows how to:
1. Set up a Curator LLM with AWS Bedrock as the backend
2. Use different Bedrock models (Claude 3.5 v2, Llama 3.3, Nova Pro, etc.)
3. Make simple prompt requests
4. Use structured chat messages
5. Create custom LLM subclasses with structured outputs
6. Compare responses from different models

To run this example, you need:
- AWS credentials with access to Bedrock
- Access to the models used in the example
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from bespokelabs import curator


# Pydantic models for structured outputs
class Recommendation(BaseModel):
    """A product recommendation."""

    product_name: str = Field(description="Name of the recommended product")
    price_range: str = Field(description="Approximate price range (e.g., '$10-20')")
    key_features: List[str] = Field(description="List of key features or benefits")
    target_audience: str = Field(description="Who this product is best for")
    rating: float = Field(description="Rating from 1-5", ge=1.0, le=5.0)


class Recommendations(BaseModel):
    """A list of product recommendations."""

    recommendations: List[Recommendation] = Field(description="List of product recommendations")


class ProductRecommender(curator.LLM):
    """Custom LLM subclass that generates product recommendations."""

    response_format = Recommendations

    def prompt(self, input: Dict[str, Any]) -> str:
        """Generate a prompt for the product recommender."""
        category = input.get("category", "tech gadgets")
        budget = input.get("budget", "$100")
        preferences = input.get("preferences", "")

        return f"""
        Recommend 3 products in the category of {category} with a budget of around {budget}.
        {preferences}

        Provide detailed recommendations including product name, price range, key features,
        target audience, and a rating from 1-5.
        """

    def parse(self, input: Dict[str, Any], response: Recommendations) -> Dict[str, Any]:
        """Parse the model response into the desired output format."""
        return {
            "category": input.get("category", "tech gadgets"),
            "budget": input.get("budget", "$100"),
            "recommendations": [
                {
                    "product_name": rec.product_name,
                    "price_range": rec.price_range,
                    "key_features": rec.key_features,
                    "target_audience": rec.target_audience,
                    "rating": rec.rating
                }
                for rec in response.recommendations
            ]
        }


def simple_prompt_example(model_name: str) -> None:
    """Example of using a simple prompt with Bedrock.

    Args:
        model_name: The Bedrock model to use
    """
    print(f"\n=== Simple Prompt Example with {model_name} ===\n")

    # Create an LLM instance with Bedrock backend
    llm = curator.LLM(
        model_name=model_name,
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Generate a response to a simple prompt
    prompt = "Explain the concept of foundation models in AI in simple terms."
    response = llm(prompt)

    print(f"Prompt: {prompt}\n")
    print(f"Response:\n{response[0]['response']}\n")


def chat_messages_example(model_name: str) -> None:
    """Example of using chat messages with Bedrock.

    Args:
        model_name: The Bedrock model to use
    """
    print(f"\n=== Chat Messages Example with {model_name} ===\n")

    # Create an LLM instance with Bedrock backend
    llm = curator.LLM(
        model_name=model_name,
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 500}
    )

    # Create chat messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that specializes in explaining complex topics simply."
        },
        {
            "role": "user",
            "content": "What is AWS Bedrock and how does it relate to foundation models?"
        }
    ]

    # Generate a response
    response = llm([messages])

    print(f"Messages: {json.dumps(messages, indent=2)}\n")
    print(f"Response:\n{response[0]['response']}\n")


def structured_output_example(model_name: str) -> None:
    """Example of using structured outputs with Bedrock.

    Args:
        model_name: The Bedrock model to use
    """
    print(f"\n=== Structured Output Example with {model_name} ===\n")

    # Create a ProductRecommender instance with Bedrock backend
    recommender = ProductRecommender(
        model_name=model_name,
        backend="bedrock",
        generation_params={"temperature": 0.7, "max_tokens": 1000}
    )

    # Generate recommendations
    input_data = {
        "category": "smart home devices",
        "budget": "$150",
        "preferences": "Energy efficient and easy to set up. Compatible with both Alexa and Google Home."
    }

    print(f"Input: {json.dumps(input_data, indent=2)}\n")

    # Generate recommendations
    recommendations = recommender(input_data)

    # Print the recommendations
    print("Recommendations:")
    for i, rec in enumerate(recommendations["recommendations"], 1):
        print(f"\n{i}. {rec['product_name']} ({rec['price_range']})")
        print(f"   Rating: {rec['rating']}/5")
        print(f"   Target Audience: {rec['target_audience']}")
        print(f"   Key Features:")
        for feature in rec['key_features']:
            print(f"   - {feature}")


def model_comparison_example(region_name: str) -> None:
    """Example comparing responses from different Bedrock models.

    Args:
        region_name: AWS region to use
    """
    print("\n=== Model Comparison Example ===\n")

    # Define models to compare
    models = [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 3.5 Sonnet v2
        "meta.llama3-3-70b-instruct-v1:0",            # Llama 3.3 70B
        "amazon.nova-pro-v1:0",                       # Amazon Nova Pro
        "mistral.mistral-large-2407-v1:0"             # Mistral Large
    ]

    # Create LLM instances for each model
    llms = {}
    for model in models:
        try:
            llms[model] = curator.LLM(
                model_name=model,
                backend="bedrock",
                backend_params={"region_name": region_name},
                generation_params={"temperature": 0.7, "max_tokens": 500}
            )
            print(f"Successfully initialized {model}")
        except Exception as e:
            print(f"Failed to initialize {model}: {str(e)}")

    if not llms:
        print("No models could be initialized. Please check your AWS Bedrock access.")
        return

    # Define a prompt for comparison
    prompt = "What are the key considerations when implementing a responsible AI system?"
    print(f"\nPrompt: {prompt}\n")

    # Generate and compare responses
    for model_name, llm in llms.items():
        try:
            print(f"\n--- {model_name} Response ---")
            response = llm(prompt)
            print(response[0]['response'])
        except Exception as e:
            print(f"Error generating response with {model_name}: {str(e)}")


def main():
    """Run the Bedrock online examples."""
    parser = argparse.ArgumentParser(description="Curator AWS Bedrock Online Examples")
    parser.add_argument("--region", type=str, default="us-west-2",
                        help="AWS region to use")
    parser.add_argument("--example", type=str, 
                        choices=["simple", "chat", "structured", "comparison", "all"],
                        default="all", help="Which example to run")
    parser.add_argument("--model", type=str, 
                        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
                        help="Bedrock model to use for single-model examples")

    args = parser.parse_args()

    # Set AWS region if provided
    if args.region:
        os.environ["AWS_REGION"] = args.region

    # Import boto3 to use the default credential provider chain
    import boto3

    # Create a boto3 session to get credentials
    session = boto3.Session(region_name=args.region)
    credentials = session.get_credentials()

    # Check if AWS credentials are available
    if not credentials:
        print("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return 1

    # Set credentials as environment variables for Curator
    if hasattr(credentials, 'access_key') and hasattr(credentials, 'secret_key'):
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        if hasattr(credentials, 'token') and credentials.token:
            os.environ["AWS_SESSION_TOKEN"] = credentials.token

    # Run the selected example(s)
    if args.example in ["simple", "all"]:
        simple_prompt_example(args.model)

    if args.example in ["chat", "all"]:
        chat_messages_example(args.model)

    if args.example in ["structured", "all"]:
        structured_output_example(args.model)

    if args.example in ["comparison", "all"]:
        model_comparison_example(args.region)

    return 0


if __name__ == "__main__":
    exit(main())
