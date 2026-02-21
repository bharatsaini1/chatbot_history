import sqlite3

# Connect to SQLite
connection = sqlite3.connect("student.db")
cursor = connection.cursor()

# Create table safely
table_info = """
CREATE TABLE Student(
    Name VARCHAR(25),
    Class VARCHAR(25),
    Section VARCHAR(25),
    Marks INT
);
"""

cursor.execute(table_info)

# Insert records
cursor.execute("INSERT INTO Student VALUES ('Bharat', 'AI Engineer', 'A', 90)")
cursor.execute("INSERT INTO Student VALUES ('John', 'ML Engineer', 'B', 80)")
cursor.execute("INSERT INTO Student VALUES ('Mukesh', 'DevOps', 'A', 70)")
cursor.execute("INSERT INTO Student VALUES ('Rupali', 'Data Science', 'B', 85)")

# Commit changes
connection.commit()

# Display records
print("The inserted records are:")
data = cursor.execute("SELECT * FROM Student")

for row in data:
    print(row)
