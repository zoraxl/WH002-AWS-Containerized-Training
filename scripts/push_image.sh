#!/bin/bash

aws --version

aws ecr get-login-password \
    --region $AWS_REGION \
| docker login \
    --username AWS \
    --password-stdin $AWS_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr create-repository \
    --repository-name aws-train \
    --image-tag-mutability IMMUTABLE \

REPO_URI=$AWS_ID.dkr.ecr.$AWS_REGION.amazonaws.com/aws-train
docker build -f ./Dockerfile -t $REPO_URI:latest .
docker push $REPO_URI:latest
