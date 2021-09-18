import os

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system("mkdir -p " + str(path))
        if os.name == 'nt':
            os.system("mkdir " + str(path))

if os.name=='nt':
    os.system("pip install wget")
    import wget

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system("git clone https://github.com/tensorflow/models " + str(paths['APIMODEL_PATH']))

if os.name == 'posix':
    os.system("apt-get install protobuf -compiler")
    os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install.")

if os.name == 'nt':
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    os.system("move protoc-3.15.6-win64.zip " + str(paths['PROTOC_PATH']))
    os.system("cd " + str(paths['PROTOC_PATH']) + " && tar -xf protoc -3.15.6-win64.zip")
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
    os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.pysetup.py && python setup.py build && python setup.py install")
    os.system("cd Tensorflow/models/research/slim && pip install -e . ")

if os.name =='posix':
    os.system("wget " + str(PRETRAINED_MODEL_URL))
    os.system("mv " + str(PRETRAINED_MODEL_NAME+'.tar.gz') + str(paths['PRETRAINED_MODEL_PATH']))
    os.system("cd " + str(paths['PRETRAINED_MODEL_PATH']) + " && tar -zxvf " + str(PRETRAINED_MODEL_NAME+'.tar.gz'))
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    os.system("move " + PRETRAINED_MODEL_NAME+'.tar.gz' + " " + str(paths['PRETRAINED_MODEL_PATH']))
    os.system("cd " + str(paths['PRETRAINED_MODEL_PATH']) + " && tar -zxvf " + PRETRAINED_MODEL_NAME+'.tar.gz')


labels = [{'name': 'thumbsup', 'id': 1}, {'name': 'thumbsdown', 'id': 2}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# OPTIONAL IF RUNNING ON COLAB
# ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
# if os.path.exists(ARCHIVE_FILES):
#     os.system("tar -zxvf " + str(ARCHIVE_FILES))

if not os.path.exists(files['TF_RECORD_SCRIPT']):
        os.system("git clone https://github.com/nicknochnack/GenerateTFRecord " + str(paths['SCRIPTS_PATH']))

os.system("python " + str(files['TF_RECORD_SCRIPT']) + " -x " + os.path.join(paths['IMAGE_PATH'], 'train') + " - l " + str(files['LABELMAP']) + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
os.system("python " + str(files['TF_RECORD_SCRIPT']) + " -x " + os.path.join(paths['IMAGE_PATH'],'test') + " - l " + str(files['LABELMAP']) + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'test.record'))

if os.name == 'posix':
    os.system("cp " + os.path.join(str(paths['PRETRAINED_MODEL_PATH']), PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))
if os.name == 'nt':
    os.system("copy " + os.path.join(str(paths['PRETRAINED_MODEL_PATH']), PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

print(config)