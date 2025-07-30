import boto3
import botocore.config
import json
from botocore.exceptions import ClientError
from datetime import datetime

def blog_generate_using_bedrock(topic: str) -> str:
    prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Write a 200 word blog on the topic: {topic}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    body = {
        "prompt" : prompt,
        "max_gen_len": 300,
        "temperature": 0.7,
        "top_p": 0.9
    }

    #Call the model

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    modelId = "meta.llama3-8b-instruct-v1:0"

    try:
        response = client.invoke_model(modelId= modelId, body= json.dumps(body))

    except Exception as e:
        print("Error occurred:", str(e))
        raise e 
        
    # Decode the response body.
    model_response = json.loads(response.get('body').read()) # or you can also write response["body"].read()

    # Extract and print the response text.
    blog = model_response["generation"]

    return blog

def save_blog_details_s3(s3_key, s3_bucket, blog_content):
    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body =blog_content)
        print("Blog saved to s3")

    except Exception as e:
        print("Error when saving the blog to s3")



def lambda_handler(event, context):
    event = json.loads(event['body'])
    blogtopic = event['blogtopic']

    blog_content = blog_generate_using_bedrock(topic=blogtopic)

    if blog_content:
        current_time = datetime.now().strftime('%H%M%S')
        s3_key = f"blog-output-folder/{current_time}.txt"
        s3_bucket = "awsblog983"
        save_blog_details_s3(s3_key, s3_bucket, blog_content)
        
    else:
        print("No blog was generated.")

    return {
        "statusCode": 200,
        "body": json.dumps({"blog": blog_content})
    }
