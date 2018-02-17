import pandas as pd
from utils.rle_mask import rle_encode
from utils.custom_functions import load

if __name__ == '__main__':
    model = load(file_name='Unet_180215_ce_with_l2')
    
    sub_df = pd.read_csv('inputs/submission.csv')
    print(sub_df['img'][0])