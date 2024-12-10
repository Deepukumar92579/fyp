import pandas as pd
df=pd.read_excel('./sanskrit.xlsx')
df.drop_duplicates()
df.dropna(inplace=True)
df.shape
df.head(10)

import re
sentence = "युयुधानो विराटश्च द्रुपदश्च महारथः ॥ ९-४॥"

# Regular expression to remove Sanskrit digits (०-९), |, ||, and commas
cleaned_sentence = re.sub(r'[०१२३४५६७८९|॥,]', '', sentence)

print(cleaned_sentence)

dataset=pd.DataFrame(df)
dataset

for index, row in df.iterrows():
    df.at[index, 'Segmented Sentence'] = re.sub(r'[०१२३४५६७८९|॥,]', '', row['Segmented Sentence'])
print(df)


for index, row in df.iterrows():
    df.at[index, 'Unsegmented Sentence'] = re.sub(r'[-|,]', '', row['Unsegmented Sentence'])
    df.at[index, 'Segmented Sentence'] = re.sub(r'[-|,]', '', row['Segmented Sentence'])