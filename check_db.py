import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = "advertisement.db"

def check_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Query all records
        c.execute("SELECT * FROM demographics LIMIT 10")
        rows = c.fetchall()
        if not rows:
            logging.warning("No data in demographics table")
        else:
            logging.info("Demographics table contents:")
            for row in rows:
                logging.info(f"ID: {row[0]}, Gender: {row[1]}, Age: {row[2]}, Timestamp: {row[3]}")
        # Query counts by gender and age
        c.execute("SELECT gender, age, COUNT(*) as count FROM demographics GROUP BY gender, age")
        counts = c.fetchall()
        if not counts:
            logging.warning("No grouped data available")
        else:
            logging.info("Demographic counts:")
            for count in counts:
                logging.info(f"Gender: {count[0]}, Age: {count[1]}, Count: {count[2]}")
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")

if __name__ == "__main__":
    check_db()