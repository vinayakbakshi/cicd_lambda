# import json

# def handler(event,context):
#     return {
#         'statuscode':200,
#         'body':json.dumps('S3 image trigger event')
#     }
import urllib.parse
import boto3

s3_client = boto3.client('s3')

def handler(event, context):
    # Check if the event is an S3 event
    if 'Records' in event and len(event['Records']) > 0 and 's3' in event['Records'][0]:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        # Check if the uploaded object is in the validationimages folder
        if key.startswith('Validation Images/'):
            # Process the uploaded image here
            # For demonstration, let's copy the uploaded image to the outputimage folder
            
            # Construct the output key in the outputimage folder
            output_key = f'Output Images/{key.split("/")[-1]}'  # Assuming key is in the format 'validationimages/image.jpg'
            
            # Copy the input image to the outputimage folder in the same bucket
            s3_client.copy_object(
                Bucket=bucket,
                Key=output_key,
                CopySource={'Bucket': bucket, 'Key': key}
            )
            
            print(f"Image uploaded to validationimages folder: {key}")
            print(f"Image copied to outputimage folder: {output_key}")
        else:
            print(f"Image uploaded to a different folder: {key}")
    else:
        print("Unsupported event format or not an S3 event")