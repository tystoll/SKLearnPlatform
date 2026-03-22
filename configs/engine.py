from sqlalchemy import create_engine
from urllib.parse import quote_plus
from src.config.database import (
    DB_SERVER,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_DRIVER,
    validate_db_env,
)

def create_sql_engine(echo: bool = False):
    validate_db_env()

    user = quote_plus(DB_USER)
    pwd = quote_plus(DB_PASSWORD)
    driver = DB_DRIVER.replace(" ", "+")

    connection_string = (
        f"mssql+pyodbc://{user}:{pwd}@{DB_SERVER}/{DB_NAME}"
        f"?driver={driver}&Encrypt=yes&TrustServerCertificate=yes"
    )

    return create_engine(
        connection_string,
        future=True,
        pool_pre_ping=True,
        fast_executemany=True,
        echo=echo,
    )

'''
from src.db.engine import create_sql_engine

engine = create_sql_engine()

df.to_sql(
    "prices",
    engine,
    if_exists="append",
    index=False,
    method="multi",
)
'''
'''
user = quote_plus(USERNAME)
pwd  = quote_plus(PASSWORD)

engine = create_engine(
    f"mssql+pyodbc://{user}:{pwd}@{SERVER}/{DATABASE}"
    f"?driver={DRIVER.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=yes",
    future=True,
    pool_pre_ping=True,
    fast_executemany=True,   # big speedup for bulk insert
)'''
'''
# 1) Convert final -> rows matching SQL columns
load_df = final.copy().reset_index()  # brings Date out of index
load_df = load_df.rename(columns={"Date": "StockDate"})
load_df["Ticker"] = TICKER

# reorder to match your SQL table
load_df = load_df[[
    "Ticker", "StockDate",
    "SMA_5", "SMA_15", "SMA_35",
    "EMA", "TEMA", "RSI",
    "Momentum", "ROC", "VolumeROC", "PVP"
]]

# Make sure StockDate is a date (not datetime w/ time)
load_df["StockDate"] = pd.to_datetime(load_df["StockDate"]).dt.date

# 2) Parameterized INSERT
insert_sql = text("""
INSERT INTO dbo.StockDerivedIndicators
(
    Ticker, StockDate,
    SMA_5, SMA_15, SMA_35,
    EMA, TEMA, RSI,
    Momentum, ROC, VolumeROC, PVP
)
VALUES
(
    :Ticker, :StockDate,
    :SMA_5, :SMA_15, :SMA_35,
    :EMA, :TEMA, :RSI,
    :Momentum, :ROC, :VolumeROC, :PVP
);
""")

rows = load_df.to_dict(orient="records")

with engine.begin() as conn:
    conn.execute(insert_sql, rows)  # executemany

print(f"Inserted {len(rows):,} rows into dbo.DerivedIndicators for {TICKER}.")
'''