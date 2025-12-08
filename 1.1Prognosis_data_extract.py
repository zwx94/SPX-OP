# %%
import os
import pandas as pd
import numpy as np

# %%
def data_handle(th_age, th_pos_neg, random_seed):
    clinical_protein = pd.read_csv(current_path + '/../data/data_handle2_save_OSed/clinical_protein_imputed.csv', index_col=0)

    clinical_protein = clinical_protein.iloc[clinical_protein['OS_time'].values >= 0,:]
    
    OS_all = pd.read_csv(current_path + '/../data/data_handle2_save_OSed/OS_all.csv', index_col=0)

    OS_age = OS_all[['age']].loc[clinical_protein.index]

    df_merged = pd.concat([OS_age, clinical_protein], axis=1, join='inner')   

    clinical_label = clinical_protein['OS_time'].values

    conditions_0_1 = [
        (clinical_label == 0),                     
        (clinical_label > 0)
    ]
    choices_0_1 = [0, 1]
    label_0_1 = np.select(conditions_0_1, choices_0_1)

    df_merged.insert(0, 'label_0_1', label_0_1)

    df = df_merged.copy()

    df = df[df['age'] >= th_age]

    age_start = ((df['age'] - th_age) // 5) * 5 + th_age
    age_end = age_start + 5


    df.insert(0, 'age_stage', age_start.astype(int).astype(str) + '~' + age_end.astype(int).astype(str))
    
    print(df[['age', 'age_stage']].head())

    pos_mask = df['label_0_1'].isin([1, 2])
    df_pos = df[pos_mask].copy()
    df_neg = df[~pos_mask].copy()   # label==0

    print('number：')
    print(df['label_0_1'].value_counts())

    neg_list = []

    for age_stage, group_pos in df_pos.groupby('age_stage'):
        n_pos = len(group_pos)
        neg_group = df_neg[df_neg['age_stage'] == age_stage]

        if neg_group.empty or n_pos == 0:
            continue

        n_neg_need = n_pos * th_pos_neg

        n_neg_to_sample = min(len(neg_group), n_neg_need)

        neg_sample = neg_group.sample(n=n_neg_to_sample, random_state=42)
        neg_list.append(neg_sample)

    if len(neg_list) > 0:
        df_neg_matched = pd.concat(neg_list, axis=0)
    else:
        df_neg_matched = pd.DataFrame(columns=df.columns)

    df_balanced = pd.concat([df_pos, df_neg_matched], axis=0)

    df_balanced = df_balanced.sample(frac=1, random_state=random_seed)

    print('Number:')
    print(df_balanced['label_0_1'].value_counts())

    return df_balanced


# %%
if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    th_age = 50
    th_pos_neg = 5
    random_seed = 2025

    # %%
    df_balanced = data_handle(th_age, th_pos_neg, random_seed)

    # %%
    df_balanced.to_csv(current_path + '/data/df_balanced.csv')

# %%
