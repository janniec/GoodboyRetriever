import numpy as np
import pandas as pd
import os
from pdf2image import convert_from_path
import boto3
import sys
sys.path.append('../MyModules/')
import Passwords as ps
from trp import Document

client = boto3.client(
    aws_access_key_id=ps.AWS_ACCESS_KEY,
    aws_secret_access_key=ps.AWS_SECRET_KEY,
    aws_session_token=ps.AWS_SESSION_TOKEN,
    service_name='textract',
    region_name='us-west-2')

session = boto3.Session(
    aws_access_key_id=ps.AWS_ACCESS_KEY,
    aws_secret_access_key=ps.AWS_SECRET_KEY,
    aws_session_token=ps.AWS_SESSION_TOKEN)

s3 = session.resource('s3')

s3BucketName = ps.TEXTRACT_S3BUCKETNAME

def rename_all_filenames(subsys_folder):
    pdfs=os.listdir('{}/'.format(subsys_folder))
    print('We are OCRing {} documents'.format(len(pdfs)))
    for pdf in pdfs:
        os.rename('{}/'.format(subsys_folder)+pdf, \
                  '{}/'.format(subsys_folder)+pdf.\
                  replace(' ', '_').replace('__', '_'))
        
def convert_pdf_to_png(pdf_folder, image_folder):
    pdfs = [f for f in os.listdir('{}/'.format(subsys_folder)) if f.endswith('.pdf')]
    for pdf in pdfs:
        images = convert_from_path('{}/'.format(subsys_folder)+pdf, 400)
        for i, image in enumerate(images):
            image.save('{}/'.format(image_folder)+'{}_page{:02d}.png'.format(pdf[:-4], i+1), 'PNG')

def move_images_to_s3(image_folder, s3, s3BucketName):
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    for image in images:
        localinput = 'images/'+image
        s3output = 'GoodboyRetrieverImages/'+image
        s3.meta.client.upload_file(localinput, s3BucketName, s3output)

def ocr_as_columns(client, s3BucketName, inputname, ocr_output_folder):
    response = client.detect_document_text(
        Document={
            'S3Object': {
                'Bucket':s3BucketName,
                'Name': inputname
            }
        })
    # detect columns and print lines
    columns = []
    lines = []
    for item in response["Blocks"]:
          if item["BlockType"] == "LINE":
            column_found=False
            for index, column in enumerate(columns):
                bbox_left = item["Geometry"]["BoundingBox"]["Left"]
                bbox_right = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]
                bbox_centre = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]/2
                column_centre = column['left'] + column['right']/2

                if (bbox_centre > column['left'] and bbox_centre < column['right']) or (column_centre > bbox_left and column_centre < bbox_right):
                    #Bbox appears inside the column
                    lines.append([index, item["Text"]])
                    column_found=True
                    break
            if not column_found:
                columns.append({'left':item["Geometry"]["BoundingBox"]["Left"], 'right':item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]})
                lines.append([len(columns)-1, item["Text"]])
    # Write to a txtfile
    outputname = inputname.split('/')[-1][:-4]+'_COLUMNS.txt'
    outputfile =open('{}/'.format(ocr_output_folder)+outputname, 'w')
                   
    lines.sort(key=lambda x: x[0])
    for line in lines:
        outputfile.write(line[1]+'\n')
    outputfile.close()

def ocr_pdfs_to_txts(pdf_folder, ocr_output_folder, image_folder='image_folder'):
    rename_all_filenames(pdf_folder)
    convert_pdf_to_png(pdf_folder, image_folder)
    move_images_to_s3(image_folder, s3, s3BucketName)
    
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')] 
    for image in images:
        if image[:-4]+'_COLUMNS.txt' in os.listdir('{}/'.fomrat(ocr_output_folder)):
            pass
        else:
            ocr_as_columns(client, s3BucketName, 'GoodboyRetrieverImages/'+image, ocr_output_folder)

        
                                   