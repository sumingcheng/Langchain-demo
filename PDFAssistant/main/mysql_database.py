import mysql.connector
from mysql.connector import Error


class MySQLDatabase:
    def __init__(self, host, user, password, database):
        self.config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }

    def create_connection(self):
        """创建数据库连接"""
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except Error as e:
            print(f"数据库连接失败: {e}")
            return None

    def save_text(self, vector_id, text):
        """将文本及其索引保存到数据库中"""
        connection = self.create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute("INSERT INTO text_entries (vector_id, text_content) VALUES (%s, %s)", (vector_id, text))
                connection.commit()
            except Error as e:
                print(f"保存文本失败: {e}")
            finally:
                cursor.close()
                connection.close()

    def get_text_by_id(self, vector_id):
        """根据索引从数据库中检索文本"""
        connection = self.create_connection()
        text = None
        if connection:
            cursor = connection.cursor()
            try:
                # 将 vector_id 转换为 Python 原生的 int 类型
                vector_id_int = int(vector_id)
                cursor.execute("SELECT text_content FROM text_entries WHERE vector_id = %s", (vector_id_int,))
                result = cursor.fetchone()
                if result:
                    text = result[0]
            except Error as e:
                print(f"查询文本失败: {e}")
            finally:
                cursor.close()
                connection.close()
        return text
