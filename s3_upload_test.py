import boto3

AWS_BUCKET_NAME = "video-kyc-ai"
AWS_REGION = "ap-south-1"

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION
)

file_path = r"C:\Users\abhik\Downloads\videokyc\Video KYC\s3_upload_test.txt"
s3_file_name = "test/s3_upload_test.txt"

s3_client.upload_file(
    file_path,
    AWS_BUCKET_NAME,
    s3_file_name
)

print("File uploaded successfully!")
print(f"S3 location: s3://{AWS_BUCKET_NAME}/{s3_file_name}")