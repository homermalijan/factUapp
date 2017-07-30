import sqlite3

conn = sqlite3.connect('db/newsDb.db')
print("Succesfully connected to DB")

def create_table():
    conn.execute('''CREATE TABLE IF NOT EXISTS news
           (TITLE           TEXT PRIMARY KEY     NOT NULL,
           AUTHORS          TEXT,
           PUBLISH_DATE     NUMERIC,
           URL              TEXT UNIQUE NOT NULL,
           BODY             TEXT NOT NULL,
           LEGIT            INTEGER
           )''')
    conn.commit()

create_table()
conn.close()
