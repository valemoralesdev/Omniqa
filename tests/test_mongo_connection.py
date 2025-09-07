import unittest
from database.mongo_connection import client  # Importación normal

class TestMongoDBConnection(unittest.TestCase):
    def test_connection(self):
        """Verifica que la conexión con MongoDB sea exitosa."""
        try:
            dbs = client.list_database_names()
            self.assertIsInstance(dbs, list)  # Debe retornar una lista de bases de datos
            print("✅ Conexión a MongoDB exitosa:", dbs)
        except Exception as e:
            self.fail(f"❌ Fallo en la conexión a MongoDB: {e}")

if __name__ == "__main__":
    unittest.main()
