#!/usr/bin/env python3
"""
Setup script for AWS resources needed for Bedrock batch examples.

This script can:
1. Create an S3 bucket and IAM role for Bedrock batch processing
2. Destroy previously created resources
3. List existing resources created by this script

Usage:
  python setup_bedrock_resources.py --create
  python setup_bedrock_resources.py --destroy
  python setup_bedrock_resources.py --list
"""

import os
import json
import time
import uuid
import argparse
import boto3
from typing import Dict, List, Optional, Tuple


def create_resources(region_name: str, bucket_prefix: str = "curator-bedrock-test", role_prefix: str = "CuratorBedrockTestRole") -> Tuple[str, str, str, str]:
    """Create S3 bucket and IAM role for Bedrock batch processing.

    Args:
        region_name: AWS region to use
        bucket_prefix: Prefix for the S3 bucket name
        role_prefix: Prefix for the IAM role name

    Returns:
        Tuple[str, str, str, str]: (bucket_name, role_arn, role_name, policy_name)
    """
    print(f"Creating AWS resources in region {region_name}...")
    
    # Create a boto3 session
    session = boto3.Session(region_name=region_name)
    
    # Create S3 bucket
    s3 = session.client('s3')
    bucket_name = f"{bucket_prefix}-{uuid.uuid4().hex[:8]}"
    
    print(f"Creating S3 bucket: {bucket_name}")
    try:
        if region_name == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region_name}
            )
        print(f"Created S3 bucket: {bucket_name}")
    except Exception as e:
        print(f"Error creating S3 bucket: {str(e)}")
        return None, None, None, None
    
    # Create IAM role
    iam = session.client('iam')
    role_name = f"{role_prefix}-{uuid.uuid4().hex[:8]}"
    
    print(f"Creating IAM role: {role_name}")
    try:
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
            Description="Role for Curator Bedrock batch processing"
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
        
        # Add a tag to identify resources created by this script
        iam.tag_role(
            RoleName=role_name,
            Tags=[
                {
                    'Key': 'CreatedBy',
                    'Value': 'CuratorBedrockSetup'
                },
                {
                    'Key': 'S3Bucket',
                    'Value': bucket_name
                }
            ]
        )
        
        # Tag the S3 bucket as well
        s3.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={
                'TagSet': [
                    {
                        'Key': 'CreatedBy',
                        'Value': 'CuratorBedrockSetup'
                    },
                    {
                        'Key': 'IAMRole',
                        'Value': role_name
                    }
                ]
            }
        )
        
        # Wait for role to propagate
        print("Waiting for IAM role to propagate...")
        time.sleep(10)
        
        # Save resource info to a file for later cleanup
        save_resource_info(bucket_name, role_name, policy_name, role_arn, region_name)
        
        print("\nResource creation complete!")
        print(f"S3 Bucket: {bucket_name}")
        print(f"IAM Role ARN: {role_arn}")
        print(f"IAM Role Name: {role_name}")
        print("\nTo use these resources with the batch examples, run:")
        print(f"python bedrock_batch_examples.py --s3-bucket {bucket_name} --role-arn {role_arn} --region {region_name}")
        
        return bucket_name, role_arn, role_name, policy_name
        
    except Exception as e:
        print(f"Error creating IAM role: {str(e)}")
        # Try to clean up the bucket if role creation failed
        try:
            print(f"Cleaning up S3 bucket: {bucket_name}")
            s3.delete_bucket(Bucket=bucket_name)
        except Exception as cleanup_error:
            print(f"Error cleaning up S3 bucket: {str(cleanup_error)}")
        
        return None, None, None, None


def destroy_resources(bucket_name: Optional[str] = None, role_name: Optional[str] = None, 
                     policy_name: Optional[str] = None, region_name: str = "us-west-2") -> bool:
    """Destroy AWS resources created for Bedrock batch processing.

    Args:
        bucket_name: Name of the S3 bucket to delete
        role_name: Name of the IAM role to delete
        policy_name: Name of the IAM policy to delete
        region_name: AWS region to use

    Returns:
        bool: True if successful, False otherwise
    """
    # If no specific resources provided, try to load from saved file
    if not bucket_name and not role_name:
        resources = load_resource_info()
        if resources:
            print("Found saved resource information:")
            for i, res in enumerate(resources, 1):
                print(f"{i}. S3 Bucket: {res.get('bucket_name')}, IAM Role: {res.get('role_name')}, Region: {res.get('region_name')}")
            
            choice = input("Enter the number of the resource set to destroy (or 'all' to destroy all): ")
            if choice.lower() == 'all':
                success = True
                for res in resources:
                    success = success and destroy_resources(
                        res.get('bucket_name'), 
                        res.get('role_name'), 
                        res.get('policy_name'), 
                        res.get('region_name', region_name)
                    )
                return success
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(resources):
                        res = resources[idx]
                        return destroy_resources(
                            res.get('bucket_name'), 
                            res.get('role_name'), 
                            res.get('policy_name'), 
                            res.get('region_name', region_name)
                        )
                    else:
                        print("Invalid selection.")
                        return False
                except ValueError:
                    print("Invalid input. Please enter a number or 'all'.")
                    return False
        else:
            print("No saved resource information found.")
            return False
    
    print(f"Destroying AWS resources in region {region_name}...")
    
    # Create a boto3 session
    session = boto3.Session(region_name=region_name)
    success = True
    
    # Delete S3 bucket
    if bucket_name:
        try:
            s3 = session.client('s3')
            print(f"Deleting objects in S3 bucket: {bucket_name}")
            
            # List and delete all objects in the bucket
            try:
                paginator = s3.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket_name):
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        if objects:
                            s3.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': objects}
                            )
            except Exception as e:
                print(f"Error listing/deleting objects: {str(e)}")
                success = False
            
            # Delete the bucket
            print(f"Deleting S3 bucket: {bucket_name}")
            s3.delete_bucket(Bucket=bucket_name)
            print(f"Deleted S3 bucket: {bucket_name}")
            
        except Exception as e:
            print(f"Error deleting S3 bucket: {str(e)}")
            success = False
    
    # Delete IAM role
    if role_name:
        try:
            iam = session.client('iam')
            
            # Delete the role policy
            if policy_name:
                print(f"Deleting IAM policy: {policy_name} from role: {role_name}")
                try:
                    iam.delete_role_policy(
                        RoleName=role_name,
                        PolicyName=policy_name
                    )
                except Exception as e:
                    print(f"Error deleting role policy: {str(e)}")
                    # Try to list and delete all policies
                    try:
                        response = iam.list_role_policies(RoleName=role_name)
                        for policy in response['PolicyNames']:
                            print(f"Deleting policy: {policy}")
                            iam.delete_role_policy(
                                RoleName=role_name,
                                PolicyName=policy
                            )
                    except Exception as list_error:
                        print(f"Error listing/deleting policies: {str(list_error)}")
            
            # Delete the role
            print(f"Deleting IAM role: {role_name}")
            iam.delete_role(RoleName=role_name)
            print(f"Deleted IAM role: {role_name}")
            
        except Exception as e:
            print(f"Error deleting IAM role: {str(e)}")
            success = False
    
    # Remove from saved resources if successful
    if success:
        remove_resource_info(bucket_name, role_name)
    
    return success


def list_resources() -> None:
    """List AWS resources created by this script."""
    resources = load_resource_info()
    
    if not resources:
        print("No saved resource information found.")
        return
    
    print("\nAWS Resources for Bedrock Batch Processing:")
    print("===========================================")
    
    for i, res in enumerate(resources, 1):
        print(f"\nResource Set {i}:")
        print(f"  S3 Bucket: {res.get('bucket_name')}")
        print(f"  IAM Role ARN: {res.get('role_arn')}")
        print(f"  IAM Role Name: {res.get('role_name')}")
        print(f"  Region: {res.get('region_name')}")
        print(f"  Created: {res.get('created_at')}")
        
        # Check if resources still exist
        try:
            session = boto3.Session(region_name=res.get('region_name', 'us-west-2'))
            s3 = session.client('s3')
            iam = session.client('iam')
            
            bucket_exists = True
            role_exists = True
            
            try:
                s3.head_bucket(Bucket=res.get('bucket_name'))
            except Exception:
                bucket_exists = False
            
            try:
                iam.get_role(RoleName=res.get('role_name'))
            except Exception:
                role_exists = False
            
            print(f"  Status: {'Active' if bucket_exists and role_exists else 'Partially deleted' if bucket_exists or role_exists else 'Deleted'}")
            
        except Exception as e:
            print(f"  Status: Error checking - {str(e)}")
    
    print("\nTo use these resources with the batch examples, run:")
    print("python bedrock_batch_examples.py --s3-bucket <bucket_name> --role-arn <role_arn> --region <region_name>")


def save_resource_info(bucket_name: str, role_name: str, policy_name: str, role_arn: str, region_name: str) -> None:
    """Save resource information to a file for later cleanup.

    Args:
        bucket_name: Name of the S3 bucket
        role_name: Name of the IAM role
        policy_name: Name of the IAM policy
        role_arn: ARN of the IAM role
        region_name: AWS region
    """
    resources_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_resources.json")
    
    # Load existing resources
    resources = []
    if os.path.exists(resources_file):
        try:
            with open(resources_file, 'r') as f:
                resources = json.load(f)
        except Exception:
            resources = []
    
    # Add new resource
    resources.append({
        'bucket_name': bucket_name,
        'role_name': role_name,
        'policy_name': policy_name,
        'role_arn': role_arn,
        'region_name': region_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Save updated resources
    with open(resources_file, 'w') as f:
        json.dump(resources, f, indent=2)


def remove_resource_info(bucket_name: str, role_name: str) -> None:
    """Remove resource information from the saved file.

    Args:
        bucket_name: Name of the S3 bucket
        role_name: Name of the IAM role
    """
    resources_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_resources.json")
    
    if not os.path.exists(resources_file):
        return
    
    # Load existing resources
    try:
        with open(resources_file, 'r') as f:
            resources = json.load(f)
    except Exception:
        return
    
    # Filter out the resource to remove
    resources = [r for r in resources if r.get('bucket_name') != bucket_name and r.get('role_name') != role_name]
    
    # Save updated resources
    with open(resources_file, 'w') as f:
        json.dump(resources, f, indent=2)


def load_resource_info() -> List[Dict]:
    """Load saved resource information.

    Returns:
        List[Dict]: List of resource information dictionaries
    """
    resources_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_resources.json")
    
    if not os.path.exists(resources_file):
        return []
    
    try:
        with open(resources_file, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup AWS resources for Bedrock batch examples")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create S3 bucket and IAM role")
    group.add_argument("--destroy", action="store_true", help="Destroy previously created resources")
    group.add_argument("--list", action="store_true", help="List existing resources")
    
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region to use")
    parser.add_argument("--bucket", type=str, help="S3 bucket name (for destroy)")
    parser.add_argument("--role", type=str, help="IAM role name (for destroy)")
    
    args = parser.parse_args()
    
    if args.create:
        create_resources(args.region)
    elif args.destroy:
        destroy_resources(args.bucket, args.role, None, args.region)
    elif args.list:
        list_resources()
    
    return 0


if __name__ == "__main__":
    exit(main())
