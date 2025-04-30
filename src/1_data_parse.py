import os 
os.environ["POLARS_FORCE_NEW_STREAMING"] = '1'
os.environ["POLARS_MAX_THREADS"] = '8'
import keyring as kr
import polars as pl
from polars import selectors as cs
import oracledb
import time
from dst import utils
import geopandas as gpd

USER = 'xw7'
PASSWORD = kr.get_password('dst', USER)

con = oracledb.connect(user=USER, password=PASSWORD, dsn='STATUDV.world')

# Query a table to polars, save as parquet and then work w/ parquet files
# Data types to save on space and computational resources: 
# - Downcast integers. 
# - Downcast pl.Datetime to pl.Date. 
# - Keep data types as they are when read from Oracle. 
#     - Seems good. Merging on integers is better than strings, because they have fixed width (32-bit)
# - Check NYKOM versions of EJER & VURK...

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MAX_RETRIES = 3

schema = {}
schema['person_id'] = pl.UInt32
schema['familie_id'] = pl.UInt32
schema['adresse_id'] = pl.UInt32
schema['kom'] = pl.UInt16
schema['reg'] = pl.UInt8
schema['cprtype'] = pl.UInt8
schema['cprtjek'] = pl.UInt8
schema['koen'] = pl.UInt8

# BEF
dataset = 'BEF'
years = [year for year in range(2008, 2023+1)]
for year in years:
    if year < 2008:
        q = f'SELECT * from D222202.{dataset}{year}12'
        df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con, infer_schema_length = None, schema_overrides = schema)
                  .filter(pl.col("person_id").is_not_null())
            )
        df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}12.pq')
    else:
        q = f'SELECT * from D222202.{dataset}{year}03'
        df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con, infer_schema_length = None, schema_overrides = schema)
                  .filter(pl.col("person_id").is_not_null())
            )
        df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}03.pq')

        q = f'SELECT * from D222202.{dataset}{year}06'
        df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con, infer_schema_length = None, schema_overrides = schema)
                  .filter(pl.col("person_id").is_not_null())
            )
        df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}06.pq')

        q = f'SELECT * from D222202.{dataset}{year}09'
        df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con, infer_schema_length = None, schema_overrides = schema)
                  .filter(pl.col("person_id").is_not_null())
            )
        df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}09.pq')

# Once finished; concat, downcast and save to .pq-file
lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True, low_memory=True).select(pl.all())
      .with_columns(cs.integer().shrink_dtype()))
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq', use_pyarrow=True)

# AKM
years = [year for year in range(1985, 2022+1)]
dataset = 'AKM'

for year in years:
    q = f'SELECT * from D222202.{dataset}{year}'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')

            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)
lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .with_columns(cs.integer().shrink_dtype()))
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# RAS
years = [year for year in range(1985, 2020+1)]
dataset = 'RAS'
for year in years:
    q = f'SELECT * from D222202.{dataset}{year}'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)
lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .with_columns(cs.integer().shrink_dtype()))
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# IND
# Don't need income in decimal/float, so first cast to Int64
years = [year for year in range(1985, 2022+1)]
dataset = 'IND'
for year in years:
    q = f'SELECT * from D222202.{dataset}{year}'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
                  .with_columns(cs.numeric().cast(pl.Int64))
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)

lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .with_columns(cs.numeric().cast(pl.Int64))
      .with_columns(cs.integer().shrink_dtype())
)
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# KRAF
years = [year for year in range(2015, 2020+1)]
dataset = 'KRAF'
for year in years:
    if year == 2014:
        pass
    q = f'SELECT * from D222202.{dataset}{year}'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)

lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .select(pl.all())
      .with_columns(cs.integer().shrink_dtype())
)
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# UDDA1
years = [year for year in range(1985, 2007+1)]
dataset = 'UDDA'
for year in years:
    q = f'SELECT * from D222202.{dataset}{year}12'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
                  .with_columns((pl.col("aar")/100).cast(pl.Int16))
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)

            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)

# UDDA2
years = [year for year in range(2008, 2020+1)]
dataset = 'UDDA'
for year in years:
    q = f'SELECT * from D222202.{dataset}{year}09'
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: 
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
                  .with_columns((pl.col("aar")/100).cast(pl.Int16))
            )
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break
        time.sleep(2)
lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .with_columns(cs.integer().shrink_dtype()))
lf.collect().write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# KOTRE
dataset = 'KOTRE'
q = f'SELECT * from D222202.{dataset}2020'
df = (pl.read_database(query=q, connection=con)
      .select(pl.all().name.to_lowercase())
      .with_columns(cs.datetime().cast(pl.Date))
      .with_columns(cs.integer().shrink_dtype()))
df.write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')
message = f'Successfully parsed {dataset}'
time_ = utils.what_time_is_it()

utils.log(message, time_)

# UDDF
dataset = 'UDDF'
q = f'SELECT * from D222202.{dataset}202209'
df = (pl.read_database(query=q, connection=con)
      .select(pl.all().name.to_lowercase())
      .with_columns((pl.col("aar")/100).cast(pl.Int32))
      .with_columns(cs.datetime().cast(pl.Date))
      .with_columns(cs.integer().shrink_dtype())
)
df.write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')
message = f'Successfully parsed {dataset}'
time_ = utils.what_time_is_it()

utils.log(message, time_)

# NETWORK
dataset = 'BOPAEL_KOORD_VER4'
q = f'SELECT * from D221916.{dataset}'

df = (pl.read_database(query=q, connection=con, infer_schema_length=None)
    .select(pl.all().name.to_lowercase())
    .with_columns(
        cs.datetime().cast(pl.Date),
        cs.integer().shrink_dtype()
    )
)
df.write_parquet(f'{DATA_DIR}/raw/bopael_koord.pq')
message = f'Successfully parsed {dataset}'
time_ = utils.what_time_is_it()

con = oracledb.connect(user=USER, password=PASSWORD, dsn='DB_GEO.world')

## Link to EJER
dataset = 'DST_EJERLINKDST_FASTEJENDOM'
q = f'SELECT * from D460207.{dataset}'
retry_count = 0
df = pl.read_database(query=q, connection=con)
df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}.pq', use_pyarrow=True)

time_ = utils.what_time_is_it()
message = f'Successfully parsed {dataset}'
print(message)
utils.log(message, time_)

# EJER IDENT
dataset = 'IDENT'

years = [year for year in range(1995, 2023+1)]
for year in years:
    q = f'SELECT * from D460207.EJDR_{dataset}{year}'
    retry_count = 0
    if year == 1998:
        continue
    elif year == 1999:
        continue
    while retry_count < MAX_RETRIES:
        try:
            df = pl.read_database(query=q, connection=con, infer_schema_length=None).select(pl.all().name.to_lowercase())
            
            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break

lf = pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_*.pq')
lf.collect().write_parquet(f'{DATA_DIR}/raw/ejdr_{dataset.lower()}.pq')

years = [year for year in range(1985, 2022+1)]

## EJER 
dataset = 'EJER'
for year in years:
    q = f'SELECT * from D460207.EJDR_{dataset}{year}'
    retry_count = 0
    if year == 2010:
        q = f'SELECT * from D460207.EJDR_{dataset}{year}_V2'
    elif year == 1987:
        continue
    while retry_count < MAX_RETRIES:
        try:
            df = pl.read_database(query=q, connection=con, infer_schema_length=None).select(pl.all().name.to_lowercase()).filter(pl.col("ejd").is_not_null()).filter(pl.col("cprsenr").is_not_null()).with_columns(cprsenr=pl.col("cprsenr").cast(pl.Int64)).with_columns(cs.numeric().cast(pl.Int64))

            df.write_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}_{year}.pq')
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break

lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq', allow_missing_columns=True)
      .select(pl.all())
      .with_columns(cs.integer().shrink_dtype())
)
lf.collect(streaming=True).write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq', use_pyarrow=True)

# VURK
dataset = 'VURK'
for year in years:
    q = f'SELECT * from D460207.EJDR_{dataset}{year}'
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            df = (utils.parse_admin_data(dataset_name=dataset, query=q, year=year, con=con)
                  .filter(pl.col("person_id").is_not_null())
                  .with_columns(aar = pl.when(pl.col("aar")>3000).then((pl.col("aar")/100).cast(pl.Int16)).otherwise(pl.col("aar")))
            )
            df.write_parquet(f'{DATA_DIR}/raw/super/raw/{dataset.lower()}_{year}.pq')
            message = f'Successfully parsed {dataset}_{year} at attempt {retry_count+1}'
            time_ = utils.what_time_is_it()

            utils.log(message, time_)
            break
        except Exception as e:
            retry_count =+ 1
            message = f'Error encountered at attempt {retry_count}. Error: {e}'
            time_ = utils.what_time_is_it()

            print(message)
            utils.log(message, time_)
            if retry_count == MAX_RETRIES:
                print(f'Could not parse {dataset}_{year} admin dataset.')    
                break

lf = (pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/{dataset.lower()}*.pq')
      .with_columns(cs.integer().shrink_dtype())
)
lf.collect(streaming=True).write_parquet(f'{DATA_DIR}/raw/{dataset.lower()}.pq')

# clusters 
gdf = gpd.read_file(f'{DATA_DIR}/raw/super_raw/clusters.shp')
gdf['cluster_id'] = gdf.index
gdf.to_parquet(f'{DATA_DIR}/raw/clusters.pq')