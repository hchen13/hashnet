db_settings = {
    'local': {
        'configs': {
            'host': '192.168.15.113',
            'username': 'ethan',
            'password': 'qwerasdf',
            'db_name': 'image_matching',
        },
        'table': 'hashnet_image'
    },
    'test': {
        'configs': {
            'host': 'rdsbuezp0btf932d168jo.mysql.rds.aliyuncs.com',
            'username': 'fishsaying',
            'password': '6uePDKEKQqT1yFKkLInY',
            'db_name': 'db_content',
        },
        'table': 'ir_upload_image'
    },
    'production': {
        'configs': {
            'host': 'rdsbuezp0btf932d168jo.mysql.rds.aliyuncs.com',
            'username': 'fishsaying',
            'password': '6uePDKEKQqT1yFKkLInY',
            'db_name': 'db_content',
        },
        'table': 'ir_upload_image'
    }
}

env = 'test'
db_params = db_settings[env]['configs']
db_tablename = db_settings[env]['table']



oss_params = {
    'key': 'LTAIbOGVE4kFjZQr',
    'secret': 'zoaB35lHB1Ph1bWKhX7hOet0B2ZNhy',
    'bucket_name': 'yushuo-image'
}
