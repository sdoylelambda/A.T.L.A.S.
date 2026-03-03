# J.A.R.V.I.S
An AI assistant based on Iron Man's J.A.R.V.I.S.

# Setup
Uses Wayland. If you are using x11 switch code in WindowController. X11 code is commented out below. Same for open_app in app_launcher. 
Wayland code is in use currently.
Create venv - 
Install requirements.txt - pip install -r requirements.txt
Run main.py - python main.py or python3 main.py

# Basic use
[x] - Speak open (program name) ie: browser - browser loads
[x] - Speak 'take a break' Jarvis does not respond until you speak 'Jarvis' or 'you there?'
[x] - Jarvis 'face', spinning orb of lights, reacts when in different states. (sleeping, listening, thinking, error)

### MVP
[x] - Voice Commands
[x] - Voice Responses
[x] - Test dummy LLM data -> Small LLM -> Larger (simply replace LLM folder with desired LLM)
[x] - Open/use Programs
Enter data
[x] - Answer general questions

### Stretch Goals
Multiple nodes
    3 Nodes setup
    1 - Brain - AI decisions - Primary Computer
    2 - Work Station - Handle AI decisions/commands - Secondary Computer
    3 - Peripherals - Remote use - Cellphone
        a. Add-ons such as cameras, temperature gauge, 

### Use Case
Using cellphone, fire up app. 
Say build me a python project that prints hello world.
Cell phone sends request to Primary Computer.
Primary computer creates a plan and sends back to the cell phone for user review. 
Cell phone either speaks plan aloud, or shows in text. (Depending on preference)
User confirms plan. Says build it.
Cell phone sends plan accepted message to Primary Computer.
Primary Computer sends plan to work station.
Work station opens Pycharm, creates project. Sends screenshot to cell phone.
User views screenshot on cellphone then says run it.
Project runs on secondary computer. (Reduces load on primary computer, periphals, monitor, etc - Primary computer free to function as normal without interference.)


Claude joint plan
Claude / Gemini API routing — infrastructure built, just needs the google-genai fix and end-to-end testing
Save window position — remember where user dragged the GUI, restore on next launch --- can't on wayland

Planned Features

GUI update — display thought process, show what Brain is doing in real time
Self-expanding keyword layer — Jarvis learns frequent commands and routes them instantly next time, skipping Brain entirely
RAG over local files — Jarvis reads your own documents and answers questions about them
Screen / vision support — LLaVA model lets Jarvis see your screen
Android client over SSH — control Jarvis from your phone
Persistent memory — remembers preferences and context across sessions
Push-to-talk mode — keyboard shortcut triggers listening, bypasses mic threshold

Known Issues Worth Fixing

Cancel mid-generation — can only cancel between steps, not while DeepSeek is writing
Double execution — occasional write_code + generate_code both firing
Whisper mishearing — upgrading to medium model would help significantly

Stretch Goals

React/Flutter GUI — full web-based interface with richer animations
Voice training — tune STT to your specific voice



i think this was already part of the plan but i want to discuss a little.

i want Jarvis to be able to remember conversations. so maybe create a md file for each conversation/project so he can
remember context and learn/grow.

it seems like i had Jarvis do a web search for time zones that it then read me the answer. how to have him always read
the answer, not just pull up a browser that has the answer?

Jarvis makes responses code always. if no code or file creation is in the request prompt, i want just an answer, not a
file with the answer.





[x] - GUI update
    [] - ui button to proceed - build it
    [x] - speaking face color
    [x] - display thought process

[] - Adapt
[] - Move files from / that aren't needed there
    [] - move actions to task folder
[] - Remove extra code/files/etc.
[] - More AI models for more tasks
    [] - General question escalation AI
    [] - Write email/paper/pdf AI
[] - fix broken tests
[] - ask what can you do? get response
    [] - say hello -> i'm jarvis...
[] - config linux/mac/windows versions
[] - top level debug? or keep per file?
[] - research {topic}
    [] - create md
    [x] - give audio summary
[x] - RULE: One sentence max for all responses. --- made it 'Keep responses as short as possible.'
[x] - test performance under load/multitasking - looks good - 1 core at 100% rest under 20%
[] - for 'bike week schedule' Jarvis should say feb 27-march 7
[] - say what can you do -> i'm jarvis, here to...
[x] - try different stt model sizes - see time delay? +5 seconds large not much better - small for fast response
[] - max recording length?
[] - read file
[] - should mistral only return code? or just when asked to create code? who should handle mid level tasks?

Context
[] - Remember Conversations
    [] - understand when a follow-up question/action is asked/requested
[] - be able to understand context and apply as appropriate
    [] - create a new {project} - now it knows we are talking about {project} until i talk about a different subject
        [] - create a new file, spreadsheet etc.
            [] - example: create a new project called gsx-r. create a price estimate for the following parts in txt file
    [] - create a new project called lead gen. make a md file for plan. execute step 1. or execute all steps in order

GUI
[] - more beams on all expressions - add till it's too much, then turn it down a little
[] - approve action button
    [] - type 'yes' to approve action

Models
[] - decide what should determine what model is used. workflow, keywords, task, hybrid, or let AI figure it out
[] - why does mistral open regular browser? is this what i want? probably better to use play write for voice controls
    [] - should mistral handle that or another model?
        [] - how can it? prompt says it can only write code.
[] Other models
    [] - Claude
    [] - analyze images
        [] - screenshot
            [] - text - summarize
                [] - text problem - how to solve - how to fix this error (or should it integrate directly into coder)
            [] - image - describe
    [] - create image
    [] - analyze PDF/document - give synapsis
    [] - what else

Bug Fix
[] - 1st word or 2 cut off - fix - debug see what is's hearing - probably filtered out. check spike duration
    [] confirm action - maybe say - confirm, please, proceed
[] - YouTube {a song} command seems to not let music play more than a few seconds. why? playwrite not playing nice?
    [] - why does google YouTube {song} work fine and not YouTube {song}

Integrations
[] - n8n integration
[] - calendar integration
    [] - what do i have today?
    [] - add event
    [] - timer --- remind me this 30 minutes to do something
    [] - speak calendar alerts
[] - send gmail
    [] - read email and respond
[] - send text
    [] - read text and respond
[] - slack alerts
[] - email alerts
[] - text alerts
[] - pycharm/vscode
    [] - line 43 create a function that does x,y,z.
    [] - go

[] - multiple commands
    [] - create stack
        [] - YouTube some music, create a file, then take a break.