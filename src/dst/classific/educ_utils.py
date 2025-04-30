import polars as pl

DATA_DIR = 'K:/Specialemappe_xw7/data'
MISC_DIR = 'K:/Specialemappe_xw7/misc'


def fetch_audd_cat() -> pl.DataFrame:
    return pl.read_csv(f'{MISC_DIR}/nomenclat_audd_ddu.csv', separator=";")

def map_audd_cats() -> pl.DataFrame:
    df_audd = fetch_audd_cat()

    df_audd = df_audd.with_columns(
        hfaudd_cat = pl.when(pl.col("KODE") == 0).then(0).when(pl.col("KODE").cast(pl.String).str.starts_with('1')).then(1)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('2')).then(2)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('31')).then(3)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('32')).then(4)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('33')).then(5)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('39')).then(5)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('5')).then(6)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('6')).then(7)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('7')).then(8)
        .when(pl.col("KODE").cast(pl.String).str.starts_with('8')).then(9)
    ).with_columns(
        years_of_edu = pl.col("hfaudd_cat").replace_strict({
            0: 10, # "Grundskole"
            1: 11, # "Forbederende uddannelser og øvrige ungdomsudd."
            2: 14, # "gym"
            3: 11, # "efu: grundforløb"
            4: 12, # "efu: studiekompetenceforløb"
            5: 14, # "efu: hovedforløb"
            6: 15, # kort videregående
            7:  16, # mellem videregående
            8: 18, # lang videregående
            9: 21 # lang videregående
        }, default = 0, return_dtype = pl.Int8)
    )

    return df_audd