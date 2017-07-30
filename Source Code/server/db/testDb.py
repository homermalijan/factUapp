import sqlite3
conn = sqlite3.connect('newsDb.db')
cur = conn.cursor()

sql = '''DELETE FROM news WHERE PUBLISH_DATE IS NULL'''
cur.execute(sql)

conn.commit()
conn.close

print("Done with database cleaning")
