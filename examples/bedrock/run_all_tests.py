#!/usr/bin/env python
"""
Run all Bedrock tests.

This script runs all the Bedrock test scripts in the examples/bedrock directory.
It automatically sets up required AWS resources (S3 bucket, IAM role) if they don't exist,
and cleans them up after the tests are complete.
"""

import os
import sys
import subprocess
import time
import json
import uuid
import argparse
import shutil
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

def print_header(message):
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def get_aws_session(profile_name=None):
    """Get an AWS session using credentials in order of priority:
    1. Already configured AWS CLI
    2. Provided AWS profile
    3. Explicit credentials from environment variables

    Args:
        profile_name: Optional AWS profile name to use

    Returns:
        boto3.Session: Configured AWS session
    """
    try:
        if profile_name:
            # Try to use the specified profile
            print(f"Using AWS profile: {profile_name}")
            return boto3.Session(profile_name=profile_name)
        else:
            # Try to use default credentials (from AWS CLI or environment variables)
            session = boto3.Session()
            if session.get_credentials():
                print("Using AWS credentials from environment or AWS CLI configuration")
                return session
            else:
                print("No AWS credentials found")
                return None
    except Exception as e:
        print(f"Error creating AWS session: {str(e)}")
        return None

def get_aws_region(session=None):
    """Get the AWS region from the AWS session or default to us-east-1."""
    if session:
        region = session.region_name
        if region:
            return region

    # Try to get region from environment variable
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region

    # Default to us-east-1
    return "us-east-1"

def create_s3_bucket(bucket_name=None, region=None, session=None):
    """Create an S3 bucket for testing if it doesn't exist."""
    if not bucket_name:
        # Generate a unique bucket name
        bucket_name = f"curator-bedrock-test-{uuid.uuid4().hex[:8]}"

    if not session:
        session = get_aws_session()
        if not session:
            print("No AWS credentials available to create S3 bucket")
            return None

    if not region:
        region = get_aws_region(session)

    print(f"Creating S3 bucket: {bucket_name} in region {region}")

    s3 = session.client('s3', region_name=region)

    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} already exists")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"Created bucket {bucket_name}")
            except Exception as e:
                print(f"Error creating bucket: {str(e)}")
                return None
        else:
            print(f"Error checking bucket: {str(e)}")
            return None

    return bucket_name

def create_iam_role(role_name=None, region=None, session=None):
    """Create an IAM role for Bedrock batch processing if it doesn't exist."""
    if not role_name:
        # Generate a unique role name
        role_name = f"CuratorBedrockTestRole-{uuid.uuid4().hex[:8]}"

    if not session:
        session = get_aws_session()
        if not session:
            print("No AWS credentials available to create IAM role")
            return None

    if not region:
        region = get_aws_region(session)

    print(f"Creating IAM role: {role_name}")

    iam = session.client('iam')

    # Define the trust policy for Bedrock
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

    try:
        # Check if role exists
        iam.get_role(RoleName=role_name)
        print(f"Role {role_name} already exists")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            # Role doesn't exist, create it
            try:
                response = iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description="Role for Curator Bedrock batch tests"
                )

                # Attach policies for S3 access
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
                )

                # Attach policies for Bedrock access
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn="arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
                )

                print(f"Created role {role_name} with necessary policies")

                # Wait for role to propagate
                print("Waiting for role to propagate...")
                time.sleep(10)

                return response['Role']['Arn']
            except Exception as e:
                print(f"Error creating role: {str(e)}")
                return None
        else:
            print(f"Error checking role: {str(e)}")
            return None

    # Get the role ARN
    try:
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']
    except Exception as e:
        print(f"Error getting role ARN: {str(e)}")
        return None

def clean_s3_bucket(bucket_name, region=None, download_dir=None, session=None):
    """Empty an S3 bucket and optionally download its contents."""
    if not session:
        session = get_aws_session()
        if not session:
            print("No AWS credentials available to clean S3 bucket")
            return False

    if not region:
        region = get_aws_region(session)

    print(f"Cleaning S3 bucket: {bucket_name}")

    s3 = session.client('s3', region_name=region)

    try:
        # List objects in the bucket
        response = s3.list_objects_v2(Bucket=bucket_name)

        if 'Contents' in response:
            # Download objects if requested
            if download_dir:
                download_path = Path(download_dir)
                download_path.mkdir(parents=True, exist_ok=True)

                print(f"Downloading bucket contents to {download_path}")

                for obj in response['Contents']:
                    key = obj['Key']
                    target_file = download_path / key

                    # Create directory structure
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    # Download file
                    s3.download_file(bucket_name, key, str(target_file))
                    print(f"Downloaded {key}")

            # Delete objects
            for obj in response['Contents']:
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                print(f"Deleted {obj['Key']}")

        print(f"Bucket {bucket_name} emptied")
        return True
    except Exception as e:
        print(f"Error cleaning bucket: {str(e)}")
        return False

def delete_s3_bucket(bucket_name, region=None, session=None):
    """Delete an S3 bucket."""
    if not session:
        session = get_aws_session()
        if not session:
            print("No AWS credentials available to delete S3 bucket")
            return False

    if not region:
        region = get_aws_region(session)

    print(f"Deleting S3 bucket: {bucket_name}")

    s3 = session.client('s3', region_name=region)

    try:
        s3.delete_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} deleted")
        return True
    except Exception as e:
        print(f"Error deleting bucket: {str(e)}")
        return False

def delete_iam_role(role_name, region=None, session=None):
    """Delete an IAM role."""
    if not session:
        session = get_aws_session()
        if not session:
            print("No AWS credentials available to delete IAM role")
            return False

    if not region:
        region = get_aws_region(session)

    print(f"Deleting IAM role: {role_name}")

    iam = session.client('iam')

    try:
        # Detach policies
        policies = [
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
        ]

        for policy in policies:
            try:
                iam.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy
                )
                print(f"Detached policy {policy}")
            except Exception as e:
                print(f"Error detaching policy {policy}: {str(e)}")

        # Delete role
        iam.delete_role(RoleName=role_name)
        print(f"Role {role_name} deleted")
        return True
    except Exception as e:
        print(f"Error deleting role: {str(e)}")
        return False

def run_test(test_script, verbose=False):
    """Run a test script and return the result.

    Args:
        test_script: Name of the test script to run
        verbose: Whether to enable verbose output

    Returns:
        bool: True if the test passed, False otherwise
    """
    print_header(f"Running {test_script}")

    # Get the full path to the script
    script_path = Path(__file__).parent / test_script

    # Set up environment for the subprocess
    env = os.environ.copy()
    env["BEDROCK_TEST_VERBOSE"] = "1" if verbose else "0"

    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env
    )

    # Print the output (always print errors, but only print stdout if verbose or there was an error)
    if verbose or result.returncode != 0:
        print(result.stdout)

    if result.stderr:
        print("ERRORS:")
        print(result.stderr)

    # Print the result
    if result.returncode == 0:
        print(f"✅ {test_script} PASSED")
    else:
        print(f"❌ {test_script} FAILED with return code {result.returncode}")

    return result.returncode == 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Bedrock tests")
    parser.add_argument("--keep-resources", action="store_true", help="Don't delete AWS resources after tests")
    parser.add_argument("--download-results", action="store_true", help="Download test results before cleanup")
    parser.add_argument("--download-dir", default="test_results", help="Directory to download results to")
    parser.add_argument("--bucket-name", help="Specify an S3 bucket name to use")
    parser.add_argument("--role-name", help="Specify an IAM role name to use")
    parser.add_argument("--profile", help="AWS profile name to use")
    parser.add_argument("--region", help="AWS region to use")
    parser.add_argument("--simple-only", action="store_true", help="Run only simple tests that don't require AWS credentials")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for test scripts")
    return parser.parse_args()

def main():
    """Run all Bedrock tests."""
    print_header("BEDROCK TESTS")

    # Parse command line arguments
    args = parse_args()

    # Get AWS session using credentials in order of priority
    aws_session = get_aws_session(args.profile)

    if not aws_session and not args.simple_only:
        print("No AWS credentials found. Running only simple tests.")
        args.simple_only = True

    # Get AWS region
    region = args.region or get_aws_region(aws_session)
    os.environ["AWS_REGION"] = region
    print(f"Using AWS region: {region}")

    # Set up AWS resources if needed
    created_resources = {}

    if not args.simple_only:
        # Create S3 bucket if needed
        bucket_name = os.environ.get("BEDROCK_BATCH_S3_BUCKET") or args.bucket_name
        if not bucket_name:
            bucket_name = create_s3_bucket(region=region, session=aws_session)
            if bucket_name:
                os.environ["BEDROCK_BATCH_S3_BUCKET"] = bucket_name
                created_resources["bucket"] = bucket_name
        else:
            # Verify bucket exists
            if create_s3_bucket(bucket_name, region, aws_session):
                os.environ["BEDROCK_BATCH_S3_BUCKET"] = bucket_name

        # Create IAM role if needed
        role_arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        if not role_arn:
            role_name = args.role_name or f"CuratorBedrockTestRole-{uuid.uuid4().hex[:8]}"
            role_arn = create_iam_role(role_name, region, aws_session)
            if role_arn:
                os.environ["BEDROCK_BATCH_ROLE_ARN"] = role_arn
                created_resources["role"] = role_name

    # Set other environment variables
    os.environ["CURATOR_DISABLE_CACHE"] = "1"
    os.environ["CURATOR_LOG_LEVEL"] = "DEBUG"

    # Check if we have all required environment variables
    required_vars = [
        "AWS_REGION",
        "BEDROCK_BATCH_S3_BUCKET",
        "BEDROCK_BATCH_ROLE_ARN"
    ]

    missing = [var for var in required_vars if not os.environ.get(var)]

    # Determine which tests to run
    if missing and not args.simple_only:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Running only simple tests that don't require AWS credentials")
        simple_only = True
    else:
        simple_only = args.simple_only

    # List of test scripts to run
    if simple_only:
        test_scripts = [
            # Simple tests that don't require actual API calls
            "test_bedrock_batch_simple.py",
        ]
    else:
        test_scripts = [
            # Simple tests
            "test_bedrock_batch_simple.py",

            # Tests that make actual API calls
            "test_bedrock_online.py",
            "test_bedrock_implementation.py",
            "test_bedrock_batch.py",
            "test_bedrock_batch_processor.py",
        ]

    # Run each test
    results = {}
    for script in test_scripts:
        results[script] = run_test(script, verbose=args.verbose)
        # Add a small delay between tests
        time.sleep(1)

    # Print summary
    print_header("TEST SUMMARY")
    for script, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {script}")

    # Clean up resources if needed
    if not args.keep_resources and created_resources:
        print_header("CLEANING UP AWS RESOURCES")

        # Clean up S3 bucket
        if "bucket" in created_resources:
            bucket_name = created_resources["bucket"]

            # Download results if requested
            if args.download_results:
                download_dir = args.download_dir
                clean_s3_bucket(bucket_name, region, download_dir, aws_session)
            else:
                clean_s3_bucket(bucket_name, region, session=aws_session)

            # Delete bucket
            delete_s3_bucket(bucket_name, region, aws_session)

        # Clean up IAM role
        if "role" in created_resources:
            role_name = created_resources["role"]
            delete_iam_role(role_name, region, aws_session)

    # Return 0 if all tests passed, 1 otherwise
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
