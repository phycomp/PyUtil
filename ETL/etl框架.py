import asyncio
from psycopg import AsyncConnection

class ETLProcessor:
    def __init__(self, db_config, fetch_batch_size, commit_batch_size):
        self.db_config = db_config
        self.fetch_batch_size = fetch_batch_size
        self.commit_batch_size = commit_batch_size
        self.db_pool = None
        self.offset = 0

    async def connect_db(self):
        if not self.db_pool:
            self.db_pool = await AsyncConnection.connect(**self.db_config)

    async def extract(self):
        query = f"SELECT * FROM source_table LIMIT {self.fetch_batch_size} OFFSET {self.offset}"
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                rows = await cur.fetchall()
        return rows

    def transform(self, data):
        return [{"id": row[0], "value": row[1].upper()} for row in data]

    async def load(self, data):
        for i in range(0, len(data), self.commit_batch_size):
            batch = data[i:i + self.commit_batch_size]
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO target_table (id, value) VALUES (%(id)s, %(value)s) ON CONFLICT (id) DO NOTHING",
                        batch
                    )
                await conn.commit()

    async def run_etl(self):
        await self.connect_db()
        while True:
            data = await self.extract()
            if not data:
                break
            transformed_data = self.transform(data)
            await self.load(transformed_data)
            self.offset += self.fetch_batch_size
