import polars as pl
import numpy as np
import random
import polars_ds as pds
from datetime import datetime, timedelta, date
import polars_graphframes as pgf

random.seed(1337)

# Params
num_rows = 100_000
id_pool = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] 
start_date = datetime(1980, 1, 1)
end_date = datetime(2020, 1, 1) 
min_days_delta = 30 
max_days_delta = 365 * 5 

# Helper func
def generate_random_dates():
    start = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    delta_days = random.randint(min_days_delta, max_days_delta)
    end = start + timedelta(days=delta_days)
    return start.date(), end.date()


seq = np.arange(0, num_rows)  # index
ids = [f"{random.choice(id_pool)}{random.randint(10, 99)}" for _ in range(num_rows)] 
dates = [generate_random_dates() for _ in range(num_rows)] 
start_dates = [d[0] for d in dates]
end_dates = [d[1] for d in dates]
# time_ranges = [timeset.TimeRange(start=d[0], end=d[1])  for d in dates]
# time_ranges = [pl.date_range(start=d[0], end=d[1]) for d in dates]
lat = np.random.uniform(-90, 90, num_rows) 
lon = np.random.uniform(-180, 180, num_rows) 

df = pl.DataFrame({
    "seq": seq,
    "id": ids,
    "start_date": start_dates,
    "end_date": end_dates,
    "lat": lat,
    "lon": lon
}, schema_overrides={'start_date': pl.Date, 'end_date': pl.Date})

print('naming cluster ids..')

df=df.with_columns(cluster_id = pgf.get_cluster_ids(
    node_definition = pl.col("seq"),
    # edge_definitions = [pl.col("start_date").over(pl.col("start_date").is_between(pl.col("start_date"),pl.col("end_date")))],
    edge_definitions = [pl.col("start_date").over(pl.col("start_date").is_between(pl.col("start_date"),pl.col("end_date"))), pl.col("end_date").over(pl.col("end_date").is_between(pl.col("start_date"),pl.col("end_date")))],
))

print('finished cluster ids..')


print(df.select(pl.col("cluster_id").n_unique()))
print(df.filter(pl.col("cluster_id")==1))