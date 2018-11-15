from db import DBManager, Image
from settings import db_params

if __name__ == '__main__':
    db = DBManager(**db_params)
    session = db.Session()
    for row in session.query(Image).all():
        row.hash = None
        row.features = None
    session.commit()
