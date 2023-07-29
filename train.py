#@title ##Input the data
# Write short explanation @markdown
import pandas as pd
from welltest.utility import *
import io

file_name = '/content/Well 4SS x Inhouse Data.xlsx'
welltest_df = get_data(file_name) #ok 'welltest/6LS Well Inhouse Data.xlsx'
welltest_df_stnd = standardize_column_names(welltest_df)
welltest_df_stnd