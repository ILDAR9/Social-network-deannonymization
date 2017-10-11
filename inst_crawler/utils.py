import itertools
import psycopg2
from InstagramAPI_diplom import InstagramAPI

def get_api(credentials):
    if not credentials:
        print("FAIL: no credentials")
        return
    attempt = 0
    for cred in itertools.cycle(credentials):
        if attempt >= 3:
            print("Can not login any user")
            break
        print(cred)
        api = InstagramAPI(*cred)
        if not api.login():
            attempt += 1
            continue
        attempt = 0
        yield api
         

class DB:
    
    def __init__(self, dbname):
        self.__dbname = dbname
        self.__create_conn(dbname)
        db = self
    
    def __create_conn(self, dbname):
        try:
            # write here your connection properties
            self.conn = psycopg2.connect("dbname='%s' user='username' host='host IP' port='5433' password='pass'" % dbname)
            self.cur = self.conn.cursor()
        except:
            print("I am unable to connect to the database")
            raise
            
    def prep_arg(self, gen):
        return b','.join(gen).decode()
        
    def execute(self, sql):
        if self.cur.closed:
            self.cur = conn.cursor()
        try:
            self.cur.execute(sql)
        except:
            self.conn.rollback()
            print('Can not execute')
            raise
        finally:
            self.conn.commit()
            
    def execute_show(self, sql):
        self.execute(sql)
        rows = self.cur.fetchall()
        for row in rows:
            print(row)
            
    def execute_get(self, sql):
        self.execute(sql)
        return self.cur.fetchall()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cur.close()
        self.conn.close()