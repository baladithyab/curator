"""Comprehensive examples demonstrating how to use Curator with AWS Bedrock for batch inference.

This example shows how to:
1. Set up a Curator LLM with AWS Bedrock as the backend for batch processing
2. Create and submit batch jobs
3. Monitor batch job status
4. Retrieve and process batch results
5. Use different Bedrock models in batch mode (Claude 3.5 Haiku, Nova Micro)
6. Create and clean up necessary AWS resources (S3 bucket, IAM role)

To run this example, you need:
- AWS credentials with access to Bedrock
- Access to the models used in the example

You can use the --create-resources flag to automatically create the necessary S3 bucket and IAM role,
or provide your own with --s3-bucket and --role-arn.
"""

import os
import json
import time
import uuid
import argparse
import asyncio
from typing import List, Dict, Any, Optional

from bespokelabs import curator


class BatchProcessor:
    """Helper class for processing batches with AWS Bedrock."""

    def __init__(
        self,
        model_name: str,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "curator-example",
        role_arn: Optional[str] = None,
        region_name: str = "us-west-2",
        timeout_hours: int = 168,
        wait: bool = False
    ):
        """Initialize the batch processor.

        Args:
            model_name: The Bedrock model to use
            s3_bucket: S3 bucket for batch data (required)
            s3_prefix: Prefix for S3 objects
            role_arn: IAM role ARN for Bedrock (required)
            region_name: AWS region to use
            timeout_hours: Maximum time in hours to wait for batch completion
            wait: Whether to wait for batch completion
        """
        if not s3_bucket or not role_arn:
            raise ValueError("S3 bucket and IAM role ARN are required for batch processing")

        self.wait = wait
        self.timeout_hours = timeout_hours

        # Set the AWS region
        os.environ["AWS_REGION"] = region_name

        # Create an LLM instance with Bedrock backend in batch mode

        # Create the config directly to ensure timeout_hours is properly set
        config = curator.request_processor.config.BatchRequestProcessorConfig(
            model=model_name,
            timeout_hours=self.timeout_hours,
            generation_params={"temperature": 0.7, "max_tokens": 300}
        )

        # Create the processor directly
        from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
        self._processor = BedrockBatchRequestProcessor(
            config=config,
            region_name=region_name,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            role_arn=role_arn
        )

        # Create the LLM properly
        self.llm = curator.LLM(
            model_name=model_name,
            backend="bedrock",
            batch=True,
            backend_params={
                "s3_bucket": s3_bucket,
                "s3_prefix": s3_prefix,
                "role_arn": role_arn,
                "region_name": region_name
            },
            generation_params={
                "temperature": 0.7,
                "max_tokens": 300
            }
        )

        self.model_name = model_name

    def process_simple_prompts(self, prompts: List[str]) -> List[str]:
        """Process a list of simple prompts in batch mode.

        Args:
            prompts: List of prompts to process

        Returns:
            List of responses
        """
        print(f"Processing {len(prompts)} prompts with {self.model_name}...")

        # Generate responses in batch mode
        dataset = self.llm(prompts)
        responses = [item.get("response", "") for item in dataset]

        # If wait is enabled, we need to monitor the batch job
        processor = self.llm._request_processor
        if self.wait and hasattr(processor, 'batch_jobs') and processor.batch_jobs:
            # Get the most recently submitted batch
            batch_id = list(processor.batch_jobs.keys())[-1]
            batch = next((b for b in processor.tracker.submitted_batches.values()
                         if b.batch_id == batch_id), None)

            if batch:
                print(f"Waiting for batch job {batch.batch_id} to complete (timeout: {self.timeout_hours} hours)...")
                try:
                    # Convert hours to seconds for the monitor_batch_job method
                    timeout_seconds = self.timeout_hours * 3600

                    # Use asyncio.run to run the coroutine in the current thread
                    import asyncio
                    batch = asyncio.run(processor.monitor_batch_job(batch, timeout_seconds))

                    if batch.status == curator.types.generic_batch.GenericBatchStatus.COMPLETE:
                        print(f"Batch job {batch.batch_id} completed successfully")
                        # Fetch the results
                        responses = asyncio.run(processor.fetch_batch_results(batch))
                    else:
                        print(f"Batch job {batch.batch_id} failed with status: {batch.status}")
                        return []
                except TimeoutError as e:
                    print(f"Batch job timed out: {str(e)}")
                    return []
                except Exception as e:
                    print(f"Error monitoring batch job: {str(e)}")
                    return []

        # Extract the response text from each response
        text_responses = []
        for response in responses:
            if isinstance(response, dict):
                text_responses.append(response.get("response", ""))
            else:
                text_responses.append(response)

        print(f"Received {len(text_responses)} responses")
        return text_responses

    def process_chat_messages(self, all_messages: List[List[Dict[str, Any]]]) -> List[str]:
        """Process a list of chat message sets in batch mode.

        Args:
            all_messages: List of chat message sets to process

        Returns:
            List of responses
        """
        print(f"Processing {len(all_messages)} chat message sets with {self.model_name}...")

        # Generate responses in batch mode
        dataset = self.llm(all_messages)
        responses = [item.get("response", "") for item in dataset]

        # If wait is enabled, we need to monitor the batch job
        processor = self.llm._request_processor
        if self.wait and hasattr(processor, 'batch_jobs') and processor.batch_jobs:
            # Get the most recently submitted batch
            batch_id = list(processor.batch_jobs.keys())[-1]
            batch = next((b for b in processor.tracker.submitted_batches.values()
                         if b.batch_id == batch_id), None)

            if batch:
                print(f"Waiting for batch job {batch.batch_id} to complete (timeout: {self.timeout_hours} hours)...")
                try:
                    # Convert hours to seconds for the monitor_batch_job method
                    timeout_seconds = self.timeout_hours * 3600

                    # Use asyncio.run to run the coroutine in the current thread
                    import asyncio
                    batch = asyncio.run(processor.monitor_batch_job(batch, timeout_seconds))

                    if batch.status == curator.types.generic_batch.GenericBatchStatus.COMPLETE:
                        print(f"Batch job {batch.batch_id} completed successfully")
                        # Fetch the results
                        responses = asyncio.run(processor.fetch_batch_results(batch))
                    else:
                        print(f"Batch job {batch.batch_id} failed with status: {batch.status}")
                        return []
                except TimeoutError as e:
                    print(f"Batch job timed out: {str(e)}")
                    return []
                except Exception as e:
                    print(f"Error monitoring batch job: {str(e)}")
                    return []

        # Extract the response text from each response
        text_responses = []
        for response in responses:
            if isinstance(response, dict):
                text_responses.append(response.get("response", ""))
            else:
                text_responses.append(response)

        print(f"Received {len(text_responses)} responses")
        return text_responses


async def simple_batch_example(
    model_name: str,
    s3_bucket: str,
    s3_prefix: str,
    role_arn: str,
    region_name: str,
    wait: bool = False,
    timeout_hours: int = 24
) -> None:
    """Example of processing simple prompts in batch mode.

    Args:
        model_name: The Bedrock model to use
        s3_bucket: S3 bucket for batch data
        s3_prefix: Prefix for S3 objects
        role_arn: IAM role ARN for Bedrock
        region_name: AWS region to use
        wait: Whether to wait for batch completion
        timeout_hours: Maximum time in hours to wait for batch completion
    """
    print(f"\n=== Simple Batch Example with {model_name} ===\n")

    # Create a batch processor
    processor = BatchProcessor(
        model_name=model_name,
        s3_bucket=s3_bucket,
        s3_prefix=f"{s3_prefix}/simple",
        role_arn=role_arn,
        region_name=region_name,
        wait=wait,
        timeout_hours=timeout_hours
    )

    # Create a list of prompts
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the key differences between supervised and unsupervised learning?",
        "How do neural networks work?",
        "What is transfer learning and why is it important?",
        "Explain the concept of overfitting and how to prevent it."
    ]

    # Process the prompts
    responses = processor.process_simple_prompts(prompts)

    # Print the results
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response {i} (first 100 chars): {response[:100]}...")


async def chat_batch_example(
    model_name: str,
    s3_bucket: str,
    s3_prefix: str,
    role_arn: str,
    region_name: str,
    wait: bool = False,
    timeout_hours: int = 24
) -> None:
    """Example of processing chat messages in batch mode.

    Args:
        model_name: The Bedrock model to use
        s3_bucket: S3 bucket for batch data
        s3_prefix: Prefix for S3 objects
        role_arn: IAM role ARN for Bedrock
        region_name: AWS region to use
        wait: Whether to wait for batch completion
        timeout_hours: Maximum time in hours to wait for batch completion
    """
    print(f"\n=== Chat Batch Example with {model_name} ===\n")

    # Create a batch processor
    processor = BatchProcessor(
        model_name=model_name,
        s3_bucket=s3_bucket,
        s3_prefix=f"{s3_prefix}/chat",
        role_arn=role_arn,
        region_name=region_name,
        wait=wait,
        timeout_hours=timeout_hours
    )

    # Create chat message sets
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
                "content": "You are a helpful AI assistant that specializes in explaining complex topics simply."
            },
            {
                "role": "user",
                "content": f"Explain {topic} to a 10-year-old."
            }
        ]
        all_messages.append(messages)

    # Process the chat messages
    responses = processor.process_chat_messages(all_messages)

    # Print the results
    for i, (topic, response) in enumerate(zip(topics, responses), 1):
        print(f"\nTopic {i}: {topic}")
        print(f"Response {i} (first 100 chars): {response[:100]}...")


async def model_comparison_example(
    s3_bucket: str,
    s3_prefix: str,
    role_arn: str,
    region_name: str,
    wait: bool = False,
    timeout_hours: int = 24
) -> None:
    """Example of comparing different models in batch mode.

    Args:
        s3_bucket: S3 bucket for batch data
        s3_prefix: Prefix for S3 objects
        role_arn: IAM role ARN for Bedrock
        region_name: AWS region to use
        wait: Whether to wait for batch completion
        timeout_hours: Maximum time in hours to wait for batch completion
    """
    print("\n=== Model Comparison Batch Example ===\n")

    # Create batch processors for different models
    claude_processor = BatchProcessor(
        model_name="anthropic.claude-3-5-haiku-20241022-v1:0",  # Claude 3.5 Haiku
        s3_bucket=s3_bucket,
        s3_prefix=f"{s3_prefix}/claude",
        role_arn=role_arn,
        region_name=region_name,
        wait=wait,
        timeout_hours=timeout_hours
    )

    nova_processor = BatchProcessor(
        model_name="amazon.nova-micro-v1:0",  # Amazon Nova Micro
        s3_bucket=s3_bucket,
        s3_prefix=f"{s3_prefix}/nova",
        role_arn=role_arn,
        region_name=region_name,
        wait=wait,
        timeout_hours=timeout_hours
    )

    # Create a list of prompts
    prompts = [
        "What is AWS Bedrock and how does it help developers?",
        "Explain the concept of foundation models in AI.",
        "What are the benefits of batch processing for AI workloads?"
    ]

    # Process the prompts with both models
    print("Submitting batch jobs to both models...")
    claude_responses = claude_processor.process_simple_prompts(prompts)
    nova_responses = nova_processor.process_simple_prompts(prompts)

    # Print the results
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Claude 3.5 Haiku Response (first 100 chars): {claude_responses[i-1][:100]}...")
        print(f"Nova Micro Response (first 100 chars): {nova_responses[i-1][:100]}...")


async def large_batch_example(
    model_name: str,
    s3_bucket: str,
    s3_prefix: str,
    role_arn: str,
    region_name: str,
    wait: bool = False,
    timeout_hours: int = 24
) -> None:
    """Example of processing a large batch of prompts.

    Args:
        model_name: The Bedrock model to use
        s3_bucket: S3 bucket for batch data
        s3_prefix: Prefix for S3 objects
        role_arn: IAM role ARN for Bedrock
        region_name: AWS region to use
        wait: Whether to wait for batch completion
        timeout_hours: Maximum time in hours to wait for batch completion
    """
    print(f"\n=== Large Batch Example with {model_name} ===\n")

    # Create a batch processor
    processor = BatchProcessor(
        model_name=model_name,
        s3_bucket=s3_bucket,
        s3_prefix=f"{s3_prefix}/large",
        role_arn=role_arn,
        region_name=region_name,
        wait=wait,
        timeout_hours=timeout_hours
    )

    # Generate a large batch of prompts (100 is the minimum for Bedrock batch)
    print("Generating 100 prompts for batch processing...")
    prompts = []
    for i in range(1, 101):
        topic = f"topic_{i % 10}"  # Cycle through 10 different topics
        prompts.append(f"Write a short paragraph about {topic}.")

    # Process the prompts
    print(f"Processing {len(prompts)} prompts in batch mode...")
    responses = processor.process_simple_prompts(prompts)

    # Print a sample of the results
    print(f"\nReceived {len(responses)} responses. Showing first 5 samples:")
    for i in range(min(5, len(responses))):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response {i+1} (first 100 chars): {responses[i][:100]}...")


async def create_aws_resources(region_name: str) -> tuple:
    """Create S3 bucket and IAM role for batch processing.

    Args:
        region_name: AWS region to use

    Returns:
        tuple: (s3_bucket, role_arn) if successful, (None, None) if failed
    """
    print("Creating AWS resources for batch processing...")

    # Import boto3 to use the default credential provider chain
    import boto3

    # Create a boto3 session
    session = boto3.Session(region_name=region_name)

    try:
        # Create S3 bucket
        s3 = session.client('s3')
        bucket_name = f"curator-bedrock-test-{uuid.uuid4().hex[:8]}"

        print(f"Creating S3 bucket: {bucket_name}")
        if region_name == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region_name}
            )
        print(f"Created S3 bucket: {bucket_name}")

        # Create IAM role
        iam = session.client('iam')
        role_name = f"CuratorBedrockTestRole-{uuid.uuid4().hex[:8]}"

        print(f"Creating IAM role: {role_name}")
        # Create trust policy for Bedrock
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        # Create the role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Temporary role for Curator Bedrock tests"
        )

        role_arn = response['Role']['Arn']
        print(f"Created IAM role with ARN: {role_arn}")

        # Attach policies for S3 access
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ]
                }
            ]
        }

        policy_name = f"CuratorBedrockTestPolicy-{uuid.uuid4().hex[:8]}"
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document)
        )

        # Wait for role to propagate
        print("Waiting for IAM role to propagate...")
        time.sleep(10)

        return bucket_name, role_arn, role_name, policy_name

    except Exception as e:
        print(f"Error creating AWS resources: {str(e)}")
        return None, None, None, None


async def cleanup_aws_resources(s3_bucket: str, role_name: str, policy_name: str, region_name: str) -> None:
    """Clean up AWS resources created for batch processing.

    Args:
        s3_bucket: S3 bucket to delete
        role_name: IAM role name to delete
        policy_name: IAM policy name to delete
        region_name: AWS region to use
    """
    print("\nCleaning up AWS resources...")

    # Import boto3 to use the default credential provider chain
    import boto3

    # Create a boto3 session
    session = boto3.Session(region_name=region_name)

    try:
        # Delete S3 bucket contents
        s3 = session.client('s3')
        print(f"Deleting objects in S3 bucket: {s3_bucket}")

        # List and delete all objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    s3.delete_objects(
                        Bucket=s3_bucket,
                        Delete={'Objects': objects}
                    )

        # Delete the bucket
        print(f"Deleting S3 bucket: {s3_bucket}")
        s3.delete_bucket(Bucket=s3_bucket)
        print(f"Deleted S3 bucket: {s3_bucket}")

    except Exception as e:
        print(f"Error deleting S3 bucket: {str(e)}")

    try:
        # Delete IAM role
        if role_name:
            iam = session.client('iam')

            # Delete the role policy
            if policy_name:
                print(f"Deleting IAM policy: {policy_name} from role: {role_name}")
                iam.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )

            # Delete the role
            print(f"Deleting IAM role: {role_name}")
            iam.delete_role(RoleName=role_name)
            print(f"Deleted IAM role: {role_name}")

    except Exception as e:
        print(f"Error deleting IAM role: {str(e)}")


async def main():
    """Run the Bedrock batch examples."""
    parser = argparse.ArgumentParser(description="Curator AWS Bedrock Batch Examples")
    parser.add_argument("--region", type=str, default="us-west-2",
                        help="AWS region to use")
    parser.add_argument("--s3-bucket", type=str,
                        help="S3 bucket for batch data")
    parser.add_argument("--s3-prefix", type=str, default="curator-example",
                        help="Prefix for S3 objects")
    parser.add_argument("--role-arn", type=str,
                        help="IAM role ARN for Bedrock")
    parser.add_argument("--example", type=str,
                        choices=["simple", "chat", "comparison", "large", "all"],
                        default="all", help="Which example to run")
    parser.add_argument("--create-resources", action="store_true",
                        help="Create S3 bucket and IAM role if not provided")
    parser.add_argument("--destroy-resources", action="store_true",
                        help="Destroy created resources after running examples")
    parser.add_argument("--claude-model", type=str,
                        default="anthropic.claude-3-5-haiku-20241022-v1:0",
                        help="Claude model to use for examples")
    parser.add_argument("--nova-model", type=str,
                        default="amazon.nova-micro-v1:0",
                        help="Nova model to use for examples")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for batch job completion")
    parser.add_argument("--timeout", type=int, default=24,
                        help="Maximum time in hours to wait for batch completion (default: 24)")
    parser.add_argument("--resume", type=str,
                        help="Resume a previously submitted batch job by ID")
    parser.add_argument("--job-arn", type=str,
                        help="ARN of the Bedrock batch job to resume")

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

    # Create S3 bucket and IAM role if needed
    s3_bucket = args.s3_bucket
    role_arn = args.role_arn
    role_name = None
    policy_name = None

    if (not s3_bucket or not role_arn) and args.create_resources:
        s3_bucket, role_arn, role_name, policy_name = await create_aws_resources(args.region)
        if not s3_bucket or not role_arn:
            print("Failed to create required AWS resources.")
            return 1

    if not s3_bucket:
        print("S3 bucket is required for batch processing. Please provide --s3-bucket or use --create-resources.")
        return 1

    if not role_arn:
        print("IAM role ARN is required for batch processing. Please provide --role-arn or use --create-resources.")
        return 1

    # Check if we need to resume a batch job
    if args.resume:
        print(f"\n=== Resuming Batch Job {args.resume} ===\n")

        # Create a temporary processor to access the batch job
        config = curator.request_processor.config.BatchRequestProcessorConfig(
            model=args.claude_model,
            timeout_hours=args.timeout
        )

        from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
        processor = BedrockBatchRequestProcessor(
            config=config,
            region_name=args.region,
            s3_bucket=s3_bucket,
            s3_prefix=args.s3_prefix,
            role_arn=role_arn
        )

        # Create a GenericBatch object with the provided ID
        batch_id = args.resume
        job_arn = args.job_arn

        if not job_arn and not batch_id.startswith("arn:aws:"):
            print("Error: When resuming a batch job, you must provide either a full job ARN or both a batch ID and job ARN.")
            print("Use --resume <batch_id> --job-arn <job_arn>")
            return 1

        # Create a minimal batch object with the provided information
        batch = curator.types.generic_batch.GenericBatch(
            batch_id=batch_id,
            id=batch_id,
            provider_batch_id=job_arn if job_arn else batch_id,
            status=curator.types.generic_batch.GenericBatchStatus.PROCESSING,
            metadata={"job_arn": job_arn} if job_arn else {}
        )

        try:
            # First, check the current status
            batch = await processor.retrieve_batch(batch)
            if not batch:
                print(f"Error: Could not retrieve batch job {args.resume}. Check the batch ID and job ARN.")
                return 1

            print(f"Current batch status: {batch.status}")

            # If the job is still processing and wait is enabled, monitor it
            if batch.status == curator.types.generic_batch.GenericBatchStatus.PROCESSING and args.wait:
                print(f"Waiting for batch job to complete (timeout: {args.timeout} hours)...")
                batch = await processor.monitor_batch_job(batch, args.timeout * 3600)

            # If the job is complete, fetch the results
            if batch.status == curator.types.generic_batch.GenericBatchStatus.COMPLETE:
                print("Batch job completed. Fetching results...")
                responses = await processor.fetch_batch_results(batch)
                print(f"Retrieved {len(responses)} responses.")

                # Print a sample of the results
                print("\nSample responses:")
                for i, response in enumerate(responses[:5]):
                    print(f"\nResponse {i+1} (first 100 chars): {response.response[:100]}...")
            else:
                print(f"Batch job is not complete. Current status: {batch.status}")
                if batch.metadata and "failure_reason" in batch.metadata:
                    print(f"Failure reason: {batch.metadata['failure_reason']}")
        except Exception as e:
            print(f"Error resuming batch job: {str(e)}")
            return 1
    else:
        # Run the selected example(s)
        try:
            if args.example in ["simple", "all"]:
                await simple_batch_example(
                    model_name=args.claude_model,
                    s3_bucket=s3_bucket,
                    s3_prefix=args.s3_prefix,
                    role_arn=role_arn,
                    region_name=args.region,
                    wait=args.wait,
                    timeout_hours=args.timeout
                )

            if args.example in ["chat", "all"]:
                await chat_batch_example(
                    model_name=args.claude_model,
                    s3_bucket=s3_bucket,
                    s3_prefix=args.s3_prefix,
                    role_arn=role_arn,
                    region_name=args.region,
                    wait=args.wait,
                    timeout_hours=args.timeout
                )

            if args.example in ["comparison", "all"]:
                await model_comparison_example(
                    s3_bucket=s3_bucket,
                    s3_prefix=args.s3_prefix,
                    role_arn=role_arn,
                    region_name=args.region,
                    wait=args.wait,
                    timeout_hours=args.timeout
                )

            if args.example in ["large", "all"]:
                await large_batch_example(
                    model_name=args.nova_model,
                    s3_bucket=s3_bucket,
                    s3_prefix=args.s3_prefix,
                    role_arn=role_arn,
                    region_name=args.region,
                    wait=args.wait,
                    timeout_hours=args.timeout
                )
        except Exception as e:
            print(f"Error running examples: {str(e)}")

    # Clean up resources if requested
    if args.destroy_resources or (args.create_resources and not args.s3_bucket and not args.role_arn):
        await cleanup_aws_resources(s3_bucket, role_name, policy_name, args.region)

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
