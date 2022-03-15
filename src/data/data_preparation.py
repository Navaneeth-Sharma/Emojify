import pandas as pd
import glob

# ########################## Train ##################################

trn_df = pd.DataFrame(columns=['file', 'label'])

folders = glob.glob('Data/train/*')
for fol in folders:
    for fil in glob.glob(fol+'/*'):
        df_tmp = pd.DataFrame(
            [(fil, fil.split('/')[::-1][1])],
            columns=['file', 'label']
            )
        trn_df = trn_df.append(df_tmp)

trn_df.to_csv('emotionTrainData.csv', index=False)

# ############################# Validation ##########################

val_df = pd.DataFrame(columns=['file', 'label'])

folders = glob.glob('Data/test/*')
for fol in folders:
    for fil in glob.glob(fol+'/*'):
        df_tmp = pd.DataFrame(
            [(fil, fil.split('/')[::-1][1])],
            columns=['file', 'label']
            )
        val_df = val_df.append(df_tmp)

val_df.to_csv('emotionValidationData.csv', index=False)
