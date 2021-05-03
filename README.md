# VTT-Parser


The VTT Parser allows you to insert a set of Zoom transcription files (.vtt) into the data folder and generate basic insights. 

##Installation

- Clone Project
- Create a python enviornment using the requirements.txt file; [Python Virtual Enviornments](https://docs.python.org/3/library/venv.html#creating-virtual-environments) 
- Place .vtt files into the data folder in the root of the project
- Run vtt_parser.py

##Usage

###Main Menu

The main menu allows you to either select an individual file to analyze or all files in the data folder. It also allows you to exit the application

###Data Options

Once the files have been parsed the user is provided with a menu of options

- 1. Data Overview
- 2. Combine Speakers
- 3. Topic Model
- 4. Clear Data; Return to Main Menu

####Data Overview

Provides a list of all speakers found within the data sorted by the duration they spoke. It also provides their mean seconds per voice capture.

###Combine Speakers

As some speakers will join from differing Zoom accounts it was necessary to allow repeats to be combined. This option allows the user to select 2 speakers, combining their data under the name of the first selected speaker. The code does a repleacement of the second speakers name with the first in the dataframe that was created after parsing the files.

###Topic Model

Provides a grouping of 1-5 topics each containing the 10 words most associated. The user is prompted for the number of topics (1-5) they wish to generate, and it provided with the top 10 words and a visual plot of them.

###Clear Data; Return to Main Menu

Wipes all the imported files and restarts the program
