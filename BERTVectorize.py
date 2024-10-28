from bert_serving.client import BertClient
import subprocess
import os

os.environ["ZEROMQ_SOCK_TMP_DIR"] = "/tmp/"

subprocess.Popen(['nohup', 'bert-serving-start', '-model_dir', '/tmp/uncased_L-24_H-1024_A-16/'])

bc = BertClient()