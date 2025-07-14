import sqlite3
conn = sqlite3.connect("advertisement.db")
c = conn.cursor()
c.execute("SELECT * FROM demographics")
print(c.fetchall())
conn.close()