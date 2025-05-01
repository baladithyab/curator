#!/usr/bin/env python3
"""
Script to empty and delete S3 buckets that match a partial name.

This script is useful for cleaning up test buckets that may have been left behind
after running Bedrock integration tests.

Usage:
    python cleanup_s3_buckets.py --name-pattern curator-bedrock-test
    python cleanup_s3_buckets.py --name-pattern curator-bedrock-test --region us-west-2 --force
"""

import os
import sys
import argparse
import logging
import boto3
from botocore.exceptions import ClientError
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default AWS region
DEFAULT_REGION = "us-west-2"


def get_aws_region(session: Optional[boto3.Session] = None) -> str:
    """Get the AWS region from environment or session.
    
    Args:
        session: Optional boto3 session
        
    Returns:
        AWS region name
    """
    if session:
        return session.region_name or os.environ.get("AWS_REGION", DEFAULT_REGION)
    return os.environ.get("AWS_REGION", DEFAULT_REGION)


def get_aws_session(region: Optional[str] = None) -> Optional[boto3.Session]:
    """Create a boto3 session with the specified region.
    
    Args:
        region: AWS region name
        
    Returns:
        boto3 Session object or None if credentials are not available
    """
    try:
        session = boto3.Session(region_name=region)
        # Verify credentials by making a simple API call
        sts = session.client('sts')
        sts.get_caller_identity()
        return session
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        return None


def list_matching_buckets(name_pattern: str, session: Optional[boto3.Session] = None) -> List[str]:
    """List all S3 buckets that match the specified name pattern.
    
    Args:
        name_pattern: Partial name to match against bucket names
        session: Optional boto3 session
        
    Returns:
        List of matching bucket names
    """
    if not session:
        session = get_aws_session()
        if not session:
            logger.error("No AWS credentials available")
            return []
    
    try:
        s3 = session.client('s3')
        response = s3.list_buckets()
        
        matching_buckets = [
            bucket['Name'] for bucket in response['Buckets']
            if name_pattern in bucket['Name']
        ]
        
        return matching_buckets
    
    except Exception as e:
        logger.error(f"Error listing buckets: {str(e)}")
        return []


def empty_bucket(bucket_name: str, session: Optional[boto3.Session] = None) -> bool:
    """Empty an S3 bucket by deleting all objects.
    
    Args:
        bucket_name: Name of the bucket to empty
        session: Optional boto3 session
        
    Returns:
        Boolean indicating if the operation was successful
    """
    if not session:
        session = get_aws_session()
        if not session:
            logger.error("No AWS credentials available")
            return False
    
    try:
        logger.info(f"Emptying bucket: {bucket_name}")
        s3 = session.client('s3')
        
        # Use pagination to handle buckets with many objects
        paginator = s3.get_paginator('list_objects_v2')
        object_count = 0
        
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )
                    object_count += len(objects)
        
        logger.info(f"Deleted {object_count} objects from bucket {bucket_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error emptying bucket {bucket_name}: {str(e)}")
        return False


def delete_bucket(bucket_name: str, session: Optional[boto3.Session] = None) -> bool:
    """Delete an S3 bucket.
    
    Args:
        bucket_name: Name of the bucket to delete
        session: Optional boto3 session
        
    Returns:
        Boolean indicating if the operation was successful
    """
    if not session:
        session = get_aws_session()
        if not session:
            logger.error("No AWS credentials available")
            return False
    
    try:
        logger.info(f"Deleting bucket: {bucket_name}")
        s3 = session.client('s3')
        s3.delete_bucket(Bucket=bucket_name)
        logger.info(f"Successfully deleted bucket: {bucket_name}")
        return True
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'BucketNotEmpty':
            logger.error(f"Bucket {bucket_name} is not empty. Empty it first.")
        else:
            logger.error(f"Error deleting bucket {bucket_name}: {str(e)}")
        return False
    
    except Exception as e:
        logger.error(f"Error deleting bucket {bucket_name}: {str(e)}")
        return False


def cleanup_matching_buckets(name_pattern: str, force: bool = False, region: Optional[str] = None) -> None:
    """Clean up all S3 buckets matching the specified pattern.
    
    Args:
        name_pattern: Partial name to match against bucket names
        force: Skip confirmation prompt if True
        region: AWS region name
    """
    # Create AWS session
    session = get_aws_session(region)
    if not session:
        logger.error("Failed to create AWS session. Check your credentials.")
        sys.exit(1)
    
    # Get actual region being used
    region = get_aws_region(session)
    logger.info(f"Using AWS region: {region}")
    
    # List matching buckets
    matching_buckets = list_matching_buckets(name_pattern, session)
    
    if not matching_buckets:
        logger.info(f"No buckets found matching pattern: {name_pattern}")
        return
    
    logger.info(f"Found {len(matching_buckets)} buckets matching pattern: {name_pattern}")
    for bucket in matching_buckets:
        logger.info(f"  - {bucket}")
    
    # Confirm deletion if not forced
    if not force:
        confirmation = input(f"\nAre you sure you want to empty and delete these {len(matching_buckets)} buckets? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            logger.info("Operation cancelled.")
            return
    
    # Process each bucket
    success_count = 0
    for bucket_name in matching_buckets:
        # Empty the bucket first
        if empty_bucket(bucket_name, session):
            # Then delete the empty bucket
            if delete_bucket(bucket_name, session):
                success_count += 1
    
    logger.info(f"Successfully cleaned up {success_count} out of {len(matching_buckets)} buckets.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Empty and delete S3 buckets matching a partial name")
    
    parser.add_argument(
        "--name-pattern",
        required=True,
        help="Partial name to match against bucket names (e.g., 'curator-bedrock-test')"
    )
    
    parser.add_argument(
        "--region",
        default=None,
        help=f"AWS region (default: from environment or {DEFAULT_REGION})"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Run the cleanup
    cleanup_matching_buckets(args.name_pattern, args.force, args.region)


if __name__ == "__main__":
    main()
