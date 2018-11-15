import oss2

from utils import load_raw

OSS_KEY = 'LTAIbOGVE4kFjZQr'
OSS_SECRET = 'zoaB35lHB1Ph1bWKhX7hOet0B2ZNhy'
BUCKET_NAME = 'yushuo-image'


endpoint = 'http://oss-cn-hangzhou.aliyuncs.com'
auth = oss2.Auth(OSS_KEY, OSS_SECRET)
bucket = oss2.Bucket(auth, endpoint, BUCKET_NAME)

class Storage:
    endpoint = 'http://oss-cn-hangzhou.aliyuncs.com'

    def __init__(self, key, secret, bucket_name):
        self.key = key
        self.secret = secret

        print("Initializing OSS connection...")
        self.auth = oss2.Auth(key, secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, bucket_name)
        print("Intialization complete!\n")

    def load_image(self, path):
        file = self.bucket.get_object(path)
        bytes = file.read()
        return load_raw(bytes)



if __name__ == '__main__':
    path = "ethan/test/亭台楼阁/pic_002.jpg"
    storage = Storage(OSS_KEY, OSS_SECRET, BUCKET_NAME)
    image = storage.load_image(path)
