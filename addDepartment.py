from mstrio import microstrategy
import pandas as pd 
import numpy as np 

username = 'baia'
password = 'HondaHitomi100%'
base_url = 'http://54.173.135.146:8080/MicroStrategyLibrary/api'
project_name = 'Model BI'

conn = microstrategy.Connection(base_url, username, password, project_name)
conn.connect()

#dataset_id_to_update = 'A71524DE409F4066DDAB9EBAE2584728'
#dataset_id_to_update = '9636294A466750BE10E32F98F423CE52'
df_group1 = pd.read_csv('./department1.csv')
conn.create_dataset(data_frame=df_group1, dataset_name='group1', table_name='group')
#conn.update_dataset(data_frame=feature_selection_df[:count_of_feature], dataset_id=dataset_id_to_update, table_name='features', update_policy='replace')
df_group2 = pd.read_csv('./department2.csv')
conn.create_dataset(data_frame=df_group2,
                    dataset_name='group2', table_name='group')
#log out
conn.close()
