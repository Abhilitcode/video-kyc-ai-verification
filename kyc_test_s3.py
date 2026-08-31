import boto3

AWS_BUCKET_NAME = "video-kyc-ai"
AWS_REGION = "ap-south-1"

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION
)

response = s3_client.list_objects_v2(
    Bucket=AWS_BUCKET_NAME
)

print("Successfully connected to S3!")
print(response)