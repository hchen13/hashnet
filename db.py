import numpy as np
from sqlalchemy import Column, Integer, String, create_engine, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from settings import db_tablename, db_params

Base = declarative_base()


class Image(Base):
    __tablename__ = db_tablename

    id = Column(Integer, primary_key=True)
    path = Column(String(10000), nullable=False, name='image')
    hash = Column(LargeBinary)
    features = Column(LargeBinary)
    res_id = Column(Integer)

    def __repr__(self):
        return "#{}: {}".format(self.id, self.path)

    def get_hash(self):
        return np.frombuffer(self.hash, dtype='int64')

    def get_features(self):
        return np.frombuffer(self.features, dtype='float32')

    def hash_str(self):
        return "".join([str(b) for b in self.get_hash()])

    def serialize(self):
        return {
            "id": self.id,
            "path": self.path,
            "res_id": self.res_id
        }


class DBManager:

    dialect = 'mysql'
    driver = 'mysqldb'

    def __init__(self, host, username, password, db_name, port=3306):
        self.host = host
        self.username = username
        self.password = password
        self.db_name = db_name
        self.port = port
        self.Session = None

        print("Initializing database...")
        self.init_database()
        print("Initialization complete!\n")

    def db_url(self):
        url_base = '{dialect}+{driver}://{username}:{password}@{host}:{port}/{db}?charset=utf8'
        url = url_base.format(
            dialect=self.dialect, driver=self.driver,
            username=self.username, password=self.password,
            host=self.host, db=self.db_name, port=self.port
        )
        return url

    def init_database(self, echo=False):
        url = self.db_url()
        engine = create_engine(url, echo=echo)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

    def list_images(self):
        session = self.Session()
        queryset = session.query(Image).all()
        return queryset


if __name__ == '__main__':
    db = DBManager(**db_params)
    image_records = db.list_images()
    print(len(image_records))