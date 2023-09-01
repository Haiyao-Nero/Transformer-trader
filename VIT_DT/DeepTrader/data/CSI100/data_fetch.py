import numpy as np
import mysql.connector
import pandas as pd
# import requests
import multiprocessing
from sqlalchemy import create_engine
# a = pd.DataFrame(np.load("industry_classification.npy"))
engine = create_engine(
    "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host='localhost', db="cstock_1d", user='haiyao',
                                                    pw='qaz1wsx2'))
djia_stocks = pd.read_csv("./csi100_stocks.csv",sep="\t")
for code in djia_stocks.Code:
    try:
        df = pd.read_sql(f"select * from `{code}`", con=engine)
        print(code,len(df))
        # df.drop(columns=["Dividend"],inplace=True)
        df.to_csv(f"./csi100_stocks/{code}.csv")
    except:
        print("Missing stock:",code)
        continue