import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect("tracking_data.db")
cursor = conn.cursor()

# Create a table to store tracking data
cursor.execute("""
CREATE TABLE IF NOT EXISTS tracked_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    name TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database initialized successfully!")
