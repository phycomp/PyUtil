import asyncio
from psycopg import  AsyncConnection
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

DB_CONFIG = {
    "host": "localhost",
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password"
}

FETCH_BATCH_SIZE, COMMIT_BATCH_SIZE = 500, 100   # 每次提取數據的數量 每次提交的數量

class ETLHandler(RequestHandler):
    async def initialize(self, db_pool):
        self.db_pool = db_pool

    async def extract(self, offset, batch_size):
        # 從數據源（例如數據庫）批量提取數據
        query = f"SELECT * FROM source_table LIMIT {batch_size} OFFSET {offset}"
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                rows = await cur.fetchall()
        return rows

    def transform(self, data):
        # 簡單轉換邏輯
        transformed_data = [{"id": row[0], "value": row[1].upper()} for row in data]
        return transformed_data

    async def load(self, data):
        # 分批次插入數據
        for i in range(0, len(data), COMMIT_BATCH_SIZE):
            batch = data[i:i + COMMIT_BATCH_SIZE]
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO target_table (id, value) VALUES (%(id)s, %(value)s) ON CONFLICT (id) DO NOTHING",
                        batch
                    )
                await conn.commit()  # 每批次提交

    async def run_etl(self):
        offset = 0
        while True:
            # 批次提取數據
            data = await self.extract(offset, FETCH_BATCH_SIZE)
            if not data:
                break  # 無數據時結束循環
            # 轉換數據
            transformed_data = self.transform(data)
            # 批量加載數據
            await self.load(transformed_data)
            offset += FETCH_BATCH_SIZE  # 更新偏移量以獲取下一批次的數據

    async def get(self):
        await self.run_etl()
        self.write("ETL process with batch fetch and commit completed!")

async def make_app():
    db_pool = await AsyncConnection.connect(**DB_CONFIG)
    return tornado.web.Application([
        (r"/run_etl", ETLHandler, dict(db_pool=db_pool)),
    ])

if __name__ == "__main__":
    app = IOLoop.current()
    app.add_callback(make_app)
    app.start()
