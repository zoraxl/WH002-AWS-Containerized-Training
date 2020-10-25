import os
import json
import time
import boto3
from sagemaker.estimator import Estimator
from sagemaker.session import Session
import tarfile

def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)


def run(mode):

    if mode == "local":
        os.system('docker build -t aws-train .')
        estimator = Estimator(image_uri='aws-train:latest', 
                              role=role, 
                              instance_count=1, 
                              instance_type='local', 
                              output_path=local_output_path, 
                              base_job_name=project_name)

        print("Local: Start fitting ... ")
        estimator.fit(inputs=f"file://{local_data_path}",job_name=job_name)
    
      

    elif mode == "sagemaker":
        s3_output_location = 's3://{}/{}'.format(bucket_name, project_name)
        estimator = Estimator(image_uri=image_uri, 
                              role=role, 
                              instance_count=1, 
                              instance_type=instance_type, 
                              output_path=s3_output_location, 
                              base_job_name=project_name)
        print("Sagemaker: Start fitting")

        estimator.fit(inputs=data_uri,job_name=job_name)

        s3_output_uri = "s3://{}/{}/{}/output/model.tar.gz".format(bucket_name, project_name, job_name)
        s3=boto_session.client('s3')
        s3_bucket, key_name = split_s3_bucket_key(s3_output_uri)
        s3.download_file(s3_bucket,key_name, 'model.tar.gz')

    else:
        print("No supported mode found. Please specify from the following: local or sagemaker")

    my_tar = tarfile.open('model.tar.gz')
    my_tar.extractall('./model')
    my_tar.close()



if __name__ == '__main__':

    local_output_path = "file://"
    local_data_path = os.getenv("DATA_PATH")
    project_name = "housing-price-prediction"
    job_name = project_name + time.strftime("-%Y-%m-%d-%H-%M", time.gmtime())

    # Credentials
    role=os.getenv("AWS_SM_ROLE")
    aws_id=os.getenv("AWS_ID")
    region=os.getenv("AWS_REGION")
    image_uri="{}.dkr.ecr.{}.amazonaws.com/aws-train".format(aws_id, region)
    print("Training image uri:{}".format(image_uri))
    instance_type=os.getenv("AWS_DEFAULT_INSTANCE")
    bucket_name = os.getenv("AWS_BUCKET")

    # comment this out if you have a bucket already or use specific bucket
    # s3 = boto3.client('s3', region_name=region)
    # s3.create_bucket(Bucket=bucket_name,CreateBucketConfiguration={'LocationConstraint': region})

    # upload the data to sagemaker
    boto_session = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_session)
    data_uri = sm_session.upload_data(local_data_path, bucket=bucket_name, key_prefix='data', extra_args=None)
    print(data_uri)
    run(mode="sagemaker")


    


