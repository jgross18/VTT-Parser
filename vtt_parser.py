import re
from datetime import datetime
from os import walk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF
# Plotting tools
import pyLDAvis
import texthero as hero
from texthero import preprocessing

from tabulate import tabulate
from texthero import stopwords



pyLDAvis.enable_notebook()  


def parse_vtt(file):
    "Read the VTT transcripts and retain"
    with open(file) as f:
        textlines = f.readlines()
    return textlines

def aggregator(items):
    "Creates dictionaries of who spoke, for how long, and what they said in a given transcript"

    def speaker_identifier(x):
        "Identifies speaker in a given parsed text"
        names = x[1].split(':')[0]
        speech = x[1].split(':')[1]
        return names, speech

    def time_cleaner(y):
        "Turns string representations of time into the difference in timestamps to measure time spoken"

        time_pattern = re.compile(r"[0-9]{2}:[0-9]{2}:[0-9]{2}")
        times = re.findall(time_pattern, y[0])
        deltas = [datetime.strptime(i, "%H:%M:%S") for i in times]
        speaking_time = deltas[1]-deltas[0]
        return speaking_time

    values = list(zip(map(time_cleaner, items),
                      map(speaker_identifier, items)))

    return [{'speaker': i[1][0], 'speaking_time': i[0], 'speech':i[1][1]} for i in values]

def createDataFrame(files): 
    transcripts = [i[2:] for i in files]

    speaker_entries = [re.split('\n[0-9]+\n', ''.join(i)) for i in transcripts]

    cleaned_items = []
    
    for j in speaker_entries:
        cleaned_items.append([i for i in [i.split('\n')[:-1]
                                        for i in j if len(i.split('\n')[:-1]) == 2] if ':' in i[1]])

    spk = [pd.DataFrame(i).groupby(
        'speaker').sum('speaking_time') for i in map(aggregator, cleaned_items)]

    concats = pd.concat([pd.DataFrame(i)
                        for i in map(aggregator, cleaned_items)], axis=0)
    concats['speaking_time_seconds'] = concats['speaking_time'] / np.timedelta64(1, 's')
    concats['speaking_time_minutes'] = round(concats['speaking_time_seconds'] / 60, 2)
    concats = concats.drop(['speaking_time'], 1)
    print('Imported Data!\n')
    
    return concats

class DFStats(object):
    def __init__(self, df):
        self.df = df
        
    
    def speakers(self): 
        return self.df['speaker'].unique()
    
    def speech(self): 
        return self.df['speech']


    def sumAndAvg(self): 
        combinedDf = self.df[['speaker', 'speaking_time_minutes']].groupby('speaker').sum(
            'speaking_time_minutes').sort_values('speaking_time_minutes', ascending=False).reset_index()
        combinedDf['average'] = self.df[['speaker', 'speaking_time_seconds']].groupby('speaker').mean(
            'speaking_time_seconds').sort_values('speaking_time_seconds', ascending=False).reset_index()['speaking_time_seconds']
        combinedDf = combinedDf.round({'average': 2})
        return combinedDf

    def sum(self):
        "Sum of speaking time data (Minutes)"
        return self.df[['speaker', 'speaking_time_minutes']].groupby('speaker').sum(
            'speaking_time_minutes').sort_values('speaking_time_minutes', ascending=False).reset_index()

    def avg(self):
        "Average length of time speaking (Seconds)"
        return self.df[['speaker', 'speaking_time_seconds']].groupby('speaker').mean(
            'speaking_time_seconds').sort_values('speaking_time_seconds', ascending=False).reset_index()

    def most_common_words(self):
        "Most common words across all data"
        data = self.df

def analysisOptions(instancesOfSpeaking): 
    print()
    print('Data Options\n')
    print('1. Data Overview')
    print('2. Combine Speakers')
    print('3. Topic Model')
    print('4. Clear Data; Return to Main Menu')

    selection = input("Selection: ")
    while int(selection) > 4 or int(selection) < 1:
        selection = input("Invalid Selection. Input New Selection: ")

    print()

    if int(selection) == 1 :
        print(tabulate(
            DFStats(instancesOfSpeaking).sumAndAvg(),
            headers=[
                'Index',
                'Speaker', 
                'Total Minutes Talking', 
                'Mean Seconds per Voice Capture'
            ]
        ))
        analysisOptions(instancesOfSpeaking)

    if int(selection) == 2 :
        speakers = DFStats(instancesOfSpeaking).sumAndAvg()
        print(tabulate(
            speakers,
            headers=[
                'Speaker ID',
                'Speaker Name', 
                'Total Minutes Talking', 
                'Mean Seconds per Voice Capture'
            ]
        ))
        selection_1 = input("Select First Speaker ID (Name Will be Kept): ")
        while  int(selection_1) > len(speakers) or int(selection_1) < 0:
            selection_1 = input("Invalid Selection. Input New First Speaker ID Selection: ")
        
        selection_2 = input("Select Second Speaker ID (Name Will be Replaced): ")
        while  int(selection_2) > len(speakers) or int(selection_2) < 0:
            selection_2 = input("Invalid Selection. Input New Second Speaker ID Selection: ")

        instancesOfSpeaking['speaker'] = instancesOfSpeaking['speaker'].replace(speakers.iloc[int(selection_2)].speaker, speakers.iloc[int(selection_1)].speaker)

        print()
        print(speakers.iloc[int(selection_2)].speaker +  ' has been combined into ' +speakers.iloc[int(selection_1)].speaker + '. ')
        
        analysisOptions(instancesOfSpeaking)

    if int(selection) == 3 :
        
        words=DFStats(instancesOfSpeaking).speech()
              
        df = pd.DataFrame.from_dict(words)
        
        df['speech'].to_string
        
        df['clean'] = hero.clean(df['speech'])
        df['clean'] = hero.remove_stopwords(df['clean'], stopwords.DEFAULT)
                
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf_vect = TfidfVectorizer(max_df=0.05, stop_words='english')
        doc_term_matrix = tfidf_vect.fit_transform(df['clean'].values.astype('U'))
        
        from sklearn.decomposition import NMF

        nmf = NMF(n_components=3, random_state=42)
        nmf.fit(doc_term_matrix )
        
        for i,topic in enumerate(nmf.components_):
            print(f'Top 10 words for topic #{i}:')
            print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')
        

        #Topic Model Goes Here
        analysisOptions(instancesOfSpeaking)
    
    if int(selection) == 4 :
        mainMenu()

def mainMenu():
    p = Path('./data').glob('*')
    # Filename Array for Selection Options
    f = []
    for (dirpath, dirnames, filenames) in walk(Path('./data')):
        f.extend(filenames)
        break
    files = [parse_vtt(i) for i in p]
        
    
    print('Welcome to the VTT Parser. Please Select an Option\n')
    print('1. Import Single VTT File')
    print('2. Import All VTT Files')
    print('3. Exit')
    
    selection = input("Selection: ")

    while int(selection) > 3 or int(selection) < 1:
        selection = input("Invalid Selection. Input New Selection: ")

    if int(selection) == 1:
        print('Select File to Import:\n')
        for i in range(len(f)):
            print(str(i+1) + '. ' + f[i])

        fileChoice = input('Selection: ')

        while int(fileChoice) > len(f) or int(fileChoice) < 1:
            fileChoice = input("Invalid Selection. Input New Selection: ")

        print()
        print('Importing ' + f[int(fileChoice) - 1] + '...\n')
        instancesOfSpeaking = createDataFrame([files[int(fileChoice) - 1]])
        print('Total Speakers Found: ' + str(len(instancesOfSpeaking['speaker'].unique())))
        print('Time Spent Speaking: ' + str(round((instancesOfSpeaking['speaking_time_seconds'].sum() /60), 2)) + ' Minutes\n')
        analysisOptions(instancesOfSpeaking)

    if int(selection) == 2:
        print()
        print('Importing All Files...')
        instancesOfSpeaking = createDataFrame(files)

        print('Total Speakers Found: ' + str(len(instancesOfSpeaking['speaker'].unique())))
        print('Time Spent Speaking: ' + str(round((instancesOfSpeaking['speaking_time_seconds'].sum() /60), 2)) + ' Minutes\n')
        analysisOptions(instancesOfSpeaking)

    if int(selection) == 3:
        return

    print()

mainMenu()

print("Thank you for using the VTT Parser.")
