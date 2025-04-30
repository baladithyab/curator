"""Utility functions for AWS Bedrock integration tests.

This module provides common functionality used across Bedrock integration tests,
including environment checking, resource setup/teardown, and test helpers.
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Any

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REGION = "us-east-1"
DEFAULT_CLAUDE_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_LLAMA_MODEL = "meta.llama3-8b-instruct-v1:0"
DEFAULT_TIMEOUT = 600  # 10 minutes


def check_aws_credentials() -> Tuple[bool, Optional[Dict[str, str]]]:
    """Check if AWS credentials are available.
    
    Returns:
        Tuple containing:
            - Boolean indicating if credentials are available
            - Dictionary with credential info if available, None otherwise
    """
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        logger.warning(f"Missing required AWS environment variables: {', '.join(missing)}")
        return False, None
    
    # Get region, with fallback to default
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    
    # Try to create a boto3 client to verify credentials
    try:
        sts = boto3.client('sts', region_name=region)
        identity = sts.get_caller_identity()
        
        return True, {
            "account_id": identity["Account"],
            "user_id": identity["UserId"],
            "arn": identity["Arn"],
            "region": region
        }
    except Exception as e:
        logger.error(f"Error verifying AWS credentials: {str(e)}")
        return False, None


def check_bedrock_access(model_id: str = DEFAULT_CLAUDE_MODEL) -> bool:
    """Check if the current AWS credentials have access to Bedrock.
    
    Args:
        model_id: The model ID to check access for
        
    Returns:
        Boolean indicating if Bedrock access is available
    """
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    
    try:
        # Create a Bedrock client
        bedrock = boto3.client('bedrock', region_name=region)
        
        # List foundation models to check access
        response = bedrock.list_foundation_models()
        
        # Check if the specified model is available
        model_ids = [model.get("modelId") for model in response.get("modelSummaries", [])]
        if model_id in model_ids:
            logger.info(f"Confirmed access to model: {model_id}")
            return True
        else:
            logger.warning(f"Model {model_id} not found in available models")
            logger.info(f"Available models: {', '.join(model_ids[:5])}...")
            return False
            
    except Exception as e:
        logger.error(f"Error checking Bedrock access: {str(e)}")
        return False


def create_test_s3_bucket(bucket_prefix: str = "curator-bedrock-test") -> Optional[str]:
    """Create a temporary S3 bucket for testing.
    
    Args:
        bucket_prefix: Prefix for the bucket name
        
    Returns:
        Bucket name if created successfully, None otherwise
    """
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    
    # Generate a unique bucket name
    bucket_name = f"{bucket_prefix}-{uuid.uuid4().hex[:8]}"
    
    try:
        # Create S3 client
        s3 = boto3.client('s3', region_name=region)
        
        # Create the bucket
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        logger.info(f"Created S3 bucket: {bucket_name}")
        
        # Wait for bucket to be available
        waiter = s3.get_waiter('bucket_exists')
        waiter.wait(Bucket=bucket_name)
        
        return bucket_name
    
    except Exception as e:
        logger.error(f"Error creating S3 bucket: {str(e)}")
        return None


def delete_s3_bucket(bucket_name: str) -> bool:
    """Delete an S3 bucket and all its contents.
    
    Args:
        bucket_name: Name of the bucket to delete
        
    Returns:
        Boolean indicating if deletion was successful
    """
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    
    try:
        # Create S3 client
        s3 = boto3.client('s3', region_name=region)
        
        # Delete all objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                s3.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': objects}
                )
        
        # Delete the bucket
        s3.delete_bucket(Bucket=bucket_name)
        logger.info(f"Deleted S3 bucket: {bucket_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting S3 bucket: {str(e)}")
        return False


def create_test_iam_role(role_name_prefix: str = "CuratorBedrockTestRole") -> Optional[str]:
    """Create a temporary IAM role for Bedrock batch processing.
    
    Args:
        role_name_prefix: Prefix for the role name
        
    Returns:
        Role ARN if created successfully, None otherwise
    """
    # Generate a unique role name
    role_name = f"{role_name_prefix}-{uuid.uuid4().hex[:8]}"
    
    try:
        # Create IAM client
        iam = boto3.client('iam')
        
        # Create the role with trust policy for Bedrock
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
        
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Temporary role for Curator Bedrock tests"
        )
        
        role_arn = response['Role']['Arn']
        logger.info(f"Created IAM role: {role_name} with ARN: {role_arn}")
        
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
                        "arn:aws:s3:::*"
                    ]
                }
            ]
        }
        
        policy_name = f"{role_name_prefix}Policy-{uuid.uuid4().hex[:8]}"
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document)
        )
        
        # Wait for role to be available (IAM changes can be slow to propagate)
        logger.info("Waiting for IAM role to propagate...")
        time.sleep(10)
        
        return role_arn
    
    except Exception as e:
        logger.error(f"Error creating IAM role: {str(e)}")
        return None


def delete_iam_role(role_name: str) -> bool:
    """Delete an IAM role and its attached policies.
    
    Args:
        role_name: Name of the role to delete
        
    Returns:
        Boolean indicating if deletion was successful
    """
    try:
        # Create IAM client
        iam = boto3.client('iam')
        
        # Delete attached policies
        response = iam.list_role_policies(RoleName=role_name)
        for policy_name in response['PolicyNames']:
            iam.delete_role_policy(
                RoleName=role_name,
                PolicyName=policy_name
            )
        
        # Delete the role
        iam.delete_role(RoleName=role_name)
        logger.info(f"Deleted IAM role: {role_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting IAM role: {str(e)}")
        return False


def extract_role_name_from_arn(role_arn: str) -> Optional[str]:
    """Extract the role name from a role ARN.
    
    Args:
        role_arn: The ARN of the role
        
    Returns:
        Role name if extraction was successful, None otherwise
    """
    try:
        # ARN format: arn:aws:iam::<account-id>:role/<role-name>
        return role_arn.split('/')[-1]
    except Exception:
        return None


def setup_test_resources() -> Dict[str, Any]:
    """Set up all necessary resources for Bedrock tests.
    
    Returns:
        Dictionary containing resource information
    """
    resources = {}
    
    # Check AWS credentials
    aws_ok, aws_info = check_aws_credentials()
    if not aws_ok:
        logger.error("AWS credentials not available, skipping resource setup")
        return resources
    
    resources.update(aws_info)
    
    # Check Bedrock access
    bedrock_ok = check_bedrock_access()
    if not bedrock_ok:
        logger.error("Bedrock access not available, skipping resource setup")
        return resources
    
    # Create S3 bucket
    bucket_name = create_test_s3_bucket()
    if bucket_name:
        resources["s3_bucket"] = bucket_name
    
    # Create IAM role
    role_arn = create_test_iam_role()
    if role_arn:
        resources["role_arn"] = role_arn
        resources["role_name"] = extract_role_name_from_arn(role_arn)
    
    return resources


def cleanup_test_resources(resources: Dict[str, Any]) -> None:
    """Clean up all resources created for testing.
    
    Args:
        resources: Dictionary containing resource information
    """
    # Delete S3 bucket
    if "s3_bucket" in resources:
        delete_s3_bucket(resources["s3_bucket"])
    
    # Delete IAM role
    if "role_name" in resources:
        delete_iam_role(resources["role_name"])
