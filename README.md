[sagemaker official tuturial](https://docs.aws.amazon.com/sagemaker/)

# udemy-aws-machine_learning

## Section-2 Data Engineering

- S3 (Simple Storage Service) Bucket
- Storage Class
  - Standard
  - Intelligent-Tiering
  - Standard-IA (Infrequent Access)
  - One Zone-IA
  - Glacier
  - Glacier Deep Archive
- Data Encrpytion
- Kinesis
  - Kinesis streams (for real-time machine learning applications)
  - Kinesis data firehose (ingest massive data near real time)
  - Kinesis data analytics (real-time ETL/ML algorithm on streams)
  - Kinesis video stream (real-time video stream to create ML applications)
  - destination: S3, redshift, elastic search, splunk, etc
 
- Glue data catalog
  - crawler:
 

# udacity-aws-machine_learning:
https://www.udacity.com/enrollment/nd189


## course 2: introduction to machine learning

- Data Wrangler
- Ground Truth

### How to run a local Jupyter Notebook on sagemaker studio:

1. login to Amazon SageMaker AI and create a notebook instance (wait till the instance is in service)
2. start jupyterLab
3. Upload Jupyter Notebook using the `up arrow` button in jupyterLab.

amazon sagemaker example: https://github.com/aws/amazon-sagemaker-examples

### AutoGluon:

An easy way to compare performance of different ML models:

https://auto.gluon.ai/stable/index.html

### project: bike sharing

https://github.com/udacity/cd0385-project-starter

## course 3: develop ML workflow:

### resources:

https://xgboost.readthedocs.io/en/latest/parameter.html
https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html
https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html


### project and exercise:

https://github.com/NxNiki/udacity-nd009t-C2-Developing-ML-Workflow

### lambda function:

- Lambda is Amazon’s serverless compute service, which is ideal for small tasks that are frequently repeated.

- A lambda function is executed when a playload is delivered to it. A payload is a JSON object which the Lambda function can read from as an input parameter.

- lambda_handler function can return a response, as long as it can be formatted as a JSON object (dict and list).

- With Lambda, as tasks are distributed redundantly across multiple machines.


#### triger a lambda function synchronously and asynchronously.

- use aws CLI:
```
aws lambda invoke --function-name preprocess-helloblze --payload '{"s3-dataset-uri": "udacity-sagemaker-solutiondata2021/l3e1/reviews_Musical_Instruments_5.json.zip"}' response.json
```
- S3 trigger


### step functions:

- task
- state machine: a workflow (an orchestrated and repeatable pattern of activity)
- function orchestration
- branching

> step functions can be expensive to run (> 100x lambda invocations), so it suitable for complex workflow that is not highly frequently used.

  












