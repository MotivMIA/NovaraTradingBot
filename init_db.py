from dotenv import load_dotenv
load_dotenv(".env.local")
from features.database import Database
Database().initialize_db()
print("Database initialized")
