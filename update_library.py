import os

from sqlalchemy import or_
from sshtunnel import SSHTunnelForwarder

from db import DBManager, Image
from oss import Storage
from prototype import Hashnet
from settings import db_params, oss_params, env

if __name__ == '__main__':
    server = None
    if env == 'production':
        if "MACHINE_ROLE" in os.environ:
            ssh_private_key = '/home/ethan/.ssh/mac_id_rsa'
        else:
            ssh_private_key = '/Users/ethan/.ssh/id_rsa'
        server = SSHTunnelForwarder(
            ("121.41.12.158", 22),
            ssh_username='root',
            ssh_pkey=ssh_private_key,
            remote_bind_address=(
                db_params['host'],
                3306
            )
        )
        server.start()
        db_params['host'] = server.local_bind_host
        db_params['port'] = server.local_bind_port
    db = DBManager(**db_params)
    storage = Storage(**oss_params)
    hashnet = Hashnet()
    hashnet.load()

    print("Scanning database for images not being processed...")
    session = db.Session()
    results = session.query(Image).filter(
        or_(
            Image.features == None,
            Image.hash == None
        )
    ).all()
    n = len(results)
    print("Scanning complete! {} in total.\n".format(n))

    print("Processing images...")
    batch_size = 32
    for i in range(0, n, batch_size):
        print("{}/{}".format(i + 1, n))
        batch = results[i : i + batch_size]
        rows = []
        images = []
        for row in batch:
            try:
                image = storage.load_image(row.path)
            except:
                continue
            if image is None or image.ndim != 3:
                continue
            if image.shape[2] == 4:
                image = image[:, :, :3]
            images.append(image)
            rows.append(row)

        hashcodes, features = hashnet.extract_features(*images)

        for i in range(len(images)):
            hash = hashcodes[i].tobytes()
            vec = features[i].tobytes()
            row = rows[i]
            row.hash = hash
            row.features = vec

        session.commit()

    session.close()
    print("Done!\n")

    if server is not None:
        server.stop()