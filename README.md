#Introduction
The purpose of annotate_sound.py is to go through sound files and annotate the sound.
Sound is divided into 3 classe:

- Client voice
- Caretaker voice
- Environmental sound

When starting annotate_sound.py, a dialogue window shows up where one can select client (upper left part of the window). 
When the client is selected the window below is populated with file names for availabele sound tracks. To 
the right of the file name there is a sound annotation status mark. It is either Yes or No depending on if the file has 
been annotated or not. Double clicking on a filename will open Audacity 
(a sound annotation tool). If an already annotated file is clicked, there will be a label import file (Import.txt) available in the 
sound annotation directory (<base directory for sound_annotation.py>/data/annotation). In Audacity, go to Files/Import/Labels and go 
to the annotation directory and select the file. The annotation labels will show up.

To annotate a sequence, mark the part to be annotated, type ctrl-B and enter the 3 character label:

- cli - for Client
- car - for Cartaker
- env - for Environmental noise

and hit the Enter key.

When finnish annotating the labels needs to be exported. Go to File/Xxport/ExportLabels. Enter 
the filename: Label.txt and press Enter. When finished exit Audacity without saving the project.



