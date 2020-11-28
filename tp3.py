"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd

# Load movies ratings data
data = pd.read_csv('data/u.data', sep='\t',
                   names=['user id', 'item id', 'rating', 'timestamp'])

# Re-arrange know ratings in a m users X n items matrix
pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index = 'user id', columns=['item id'])

input("press any key to exit...")
