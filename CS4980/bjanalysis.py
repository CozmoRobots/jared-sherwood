#now that we have a csv file with our training data, we can do some analysis

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('blackjackdata02ML')
print(df.info())

#let's get some data
#first, let's get a breakdown of the outcomes: win: 1, lose: -1, tie: 0
print(df['result'].value_counts())

#we should create a new field to combine win/tie to differentiate between 
#that outcome and a loss
df['win_tie'] = np.where(df['result'] != -1, 1, 0)

print(df['win_tie'].value_counts())


#let's compute some probabilities based on the dealer card
data1 = (df.groupby(by='init_hand').sum()['win_tie'] / df.groupby(by='init_hand').count()['win_tie'])

fig, ax = plt.subplots(figsize = (10,6))
ax = sns.barplot(x=data1.index, y=data1.values)
ax.set_xlabel("Player's Initial Hand", fontsize=16)
ax.set_ylabel("Probability of a Win or Tie", fontsize = 16)

plt.tight_layout()
plt.savefig(fname='Initial_Hand_Probabilities', dpi=150)
plt.show()

#we could do the same for probabilities for the player's initial hand value
data2 = (df.groupby(by='dealer_card').sum()['win_tie'] / df.groupby(by='dealer_card').count()['win_tie'])

fig, ax = plt.subplots(figsize = (10,6))
ax = sns.barplot(x=data2.index, y=data2.values)
ax.set_xlabel("", fontsize=16)
ax.set_ylabel("Probability of a Win or Tie", fontsize = 16)

plt.tight_layout()
plt.savefig(fname="Dealer's Card", dpi=150)
plt.show()

#finally, let's compare our simulators
df_better = pd.read_csv('blackjackEditedData03-WithML')
print(df_better.info())

#let's get some data
#first, let's get a breakdown of the outcomes: win: 1, lose: -1, tie: 0
print(df_better['result'].value_counts())

#we should create a new field to combine win/tie to differentiate between 
#that outcome and a loss
df_better['win_tie'] = np.where(df_better['result'] != -1, 1, 0)

print(df_better['win_tie'].value_counts())


#let's compute some probabilities based on the dealer card
data_better = (df_better.groupby(by='init_hand').sum()['win_tie'] / df_better.groupby(by='init_hand').count()['win_tie'])

#put the two dataframes together
data_comp = pd.DataFrame()
data_comp['naive'] = data1
data_comp['better'] = data_better

fig, ax = plt.subplots(figsize = (10,6))
ax.bar(x=data1.index-0.2, height=data_comp['naive'].values, color='blue', width=0.4, label='Naive')
ax.bar(x=data1.index+0.2, height=data_comp['better'].values, color='red', width=0.4, label='Better')


ax.set_xlabel("Initial Hand", fontsize=16)
ax.set_ylabel("Probability of a Win or Tie", fontsize = 16)
#the initial hand COULD range from 2 to 21...the dealer's card COULD range from 2 to 11...depending on how we valued an ace
plt.xticks(np.arange(2, 22, 1.0))

plt.legend()
plt.tight_layout()
plt.savefig(fname="InitHandComp", dpi=150)
plt.show()

