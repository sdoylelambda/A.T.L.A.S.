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

### Next Steps
[] - Remember Conversations
[] - Adapt
[] - Move files from / that aren't needed there
[] - Remove extra code/files/etc.
[] - More AI models for more tasks
    [] - General question escalation AI
    [] - Write email/paper/pdf AI
[] - ask what can you do? get response
[] - config linux/mac/windows versions
[] - top level debug? or keep per file?
[] - research {topic} create md and give short audio summary
[] - RULE: One sentence max for all responses. --- make 2 sentences?
[] - calendar integration
    [] - what do i have today? add event
    [] - timer --- remind me this 30 minutes to call jon
[] - test performance under load/multitasking