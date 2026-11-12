import tkinter as tk
import tkinter.font as tkfont
import time
import math
import threading
import queue
import wave
import numpy as np
import AudioModule          # microphone recording (the original fingerprint code is kept for reference)
import EngineModule         # improved fingerprint + index + statistical decision
from collections import deque
from timeit import default_timer as timer
import pickle
import os


# ---------- Look and feel ----------
BG = '#0f1117'          # Window background
SURFACE = '#1a1e29'     # Cards
SURFACE_HOVER = '#232838'
BORDER = '#2a3040'
TEXT = '#f2f4f8'
MUTED = '#8a93a6'
ACCENT = '#4f8cff'      # Main blue
ACCENT_HOVER = '#6a9eff'
LISTEN = '#ff4d6d'      # Colour while recording
SUCCESS = '#3ddc97'
WARNING = '#ffb547'

RECORD_SECONDS = 15
HISTORY_FILE = "lastSongsMatched.txt"


def PickFont(): # Use Segoe UI on Windows, otherwise fall back to Helvetica
    return 'Segoe UI' if 'Segoe UI' in tkfont.families() else 'Helvetica'


def BlendColours(colourA, colourB, amount): # Mix two hex colours. amount=0 gives colourA, amount=1 gives colourB (tkinter has no transparency)
    a = [int(colourA[i:i+2], 16) for i in (1, 3, 5)]
    b = [int(colourB[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join('%02x' % round(a[i] + (b[i] - a[i]) * amount) for i in range(3))


class FlatButton(tk.Label): # A flat, rounded-looking button with a hover effect (tk.Button can't be styled well on Windows)
    def __init__(self, parent, text, command, bg=ACCENT, hover=ACCENT_HOVER, fg=TEXT, font=None, padx=28, pady=10):
        tk.Label.__init__(self, parent, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, cursor='hand2')
        self.normalColour, self.hoverColour, self.command = bg, hover, command
        self.bind('<Enter>', lambda e: self.config(bg=self.hoverColour))
        self.bind('<Leave>', lambda e: self.config(bg=self.normalColour))
        self.bind('<Button-1>', lambda e: self.command())


class MainApplication(tk.Frame): # Class for the main GUI of the application. Inherits methods from tk.Frame
    def __init__(self, parent, *args, **kwargs):
        # Initialise the applications main frame (always use the app's own background colour)
        kwargs['bg'] = BG
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # Setup window size and name
        self.parent.geometry("400x640")
        self.parent.minsize(360, 600)
        self.parent.title("MusicID")
        self.parent.configure(bg=BG)

        # Fonts
        family = PickFont()
        self.fonts = {
            'logo': (family, 22, 'bold'),
            'subtitle': (family, 11),
            'status': (family, 15, 'bold'),
            'small': (family, 10),
            'eyebrow': (family, 9, 'bold'),
            'songTitle': (family, 22, 'bold'),
            'artist': (family, 14),
            'meta': (family, 11),
            'button': (family, 11, 'bold'),
            'heading': (family, 16, 'bold'),
            'row': (family, 11),
            'icon': (family, 34, 'bold'),
        }

        # Specify some conditions at launch. The song library is loaded on a background thread
        self.engine = None
        self.pointerInAudio = 0
        self.listening = False
        self.LoadHistory()

        # Header that stays at the top of every screen
        header = tk.Frame(self, bg=BG)
        header.pack(side=tk.TOP, fill=tk.X, pady=(22, 0))
        self.titleLabel = tk.Label(header, text="MusicID", font=self.fonts['logo'], bg=BG, fg=TEXT)
        self.titleLabel.pack()
        self.libraryLabel = tk.Label(header, text='Loading song library…', font=self.fonts['small'], bg=BG, fg=MUTED)
        self.libraryLabel.pack()
        self.engineQueue = queue.Queue()
        threading.Thread(target=lambda: self.engineQueue.put(EngineModule.Engine()), daemon=True).start()
        self.after(100, self.CheckEngineLoaded)

        # Each screen (home, results, history) is built inside this frame and replaced when the screen changes
        self.screen = None
        self.ShowHome()

    # ---------- Screen helpers ----------
    def NewScreen(self): # Remove the current screen and return a fresh empty one
        if self.screen is not None:
            self.screen.destroy()
        self.screen = tk.Frame(self, bg=BG)
        self.screen.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=(10, 24))
        return self.screen

    def ResetWidgets(self): # Go back to the home screen, so that the user can record again
        self.ShowHome()

    def CheckEngineLoaded(self): # Runs on the GUI thread until the library has finished loading
        try:
            self.engine = self.engineQueue.get_nowait()
        except queue.Empty:
            self.after(100, self.CheckEngineLoaded)
            return
        n = self.engine.n_songs
        self.libraryLabel.config(text=f'{n:,} song' + ('' if n == 1 else 's') + ' in library')
        if not self.listening and self.statusLabel.winfo_exists():
            self.statusLabel.config(text='Tap to identify a song')

    def LoadHistory(self): # Load the queue for storing the last 10 matches (start a fresh one if the file is missing or unreadable)
        try:
            with open(os.path.join(os.getcwd(), HISTORY_FILE), "rb") as savefile:
                self.lastSongsMatched = deque(pickle.load(savefile), maxlen=10)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.lastSongsMatched = deque(maxlen=10)

    def SaveHistory(self):
        with open(os.path.join(os.getcwd(), HISTORY_FILE), "wb") as savefile:
            pickle.dump(self.lastSongsMatched, savefile)

    # ---------- Home / listening screen ----------
    def ShowHome(self):
        screen = self.NewScreen()

        # The big round button, drawn on a canvas so it can be a circle and animate
        self.canvasSize = 300
        self.canvas = tk.Canvas(screen, width=self.canvasSize, height=self.canvasSize, bg=BG, highlightthickness=0)
        self.canvas.pack(pady=(30, 0))
        self.DrawRecordButton(ACCENT)
        self.canvas.tag_bind('button', '<Button-1>', lambda e: self.recordButtonClick())
        self.canvas.tag_bind('button', '<Enter>', lambda e: self.HoverRecordButton(True))
        self.canvas.tag_bind('button', '<Leave>', lambda e: self.HoverRecordButton(False))

        self.statusLabel = tk.Label(screen, text='Tap to identify a song', font=self.fonts['status'], bg=BG, fg=TEXT)
        self.statusLabel.pack(pady=(14, 2))
        self.hintLabel = tk.Label(screen, text='Play music near your microphone', font=self.fonts['subtitle'], bg=BG, fg=MUTED)
        self.hintLabel.pack()

        # Recent matches button at the bottom
        count = len(self.lastSongsMatched)
        self.lastSongsButton = FlatButton(screen, 'Recent matches' + (f'  ·  {count}' if count else ''), self.ShowLastMatches,
                                          bg=SURFACE, hover=SURFACE_HOVER, font=self.fonts['button'])
        self.lastSongsButton.pack(side=tk.BOTTOM, pady=(0, 6))

    def DrawRecordButton(self, colour): # Draw the circular button with a microphone icon in the middle
        c, cx = self.canvas, self.canvasSize / 2
        cy = cx
        c.delete('all')
        r = 82
        c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=colour, outline='', tags=('button', 'circle'))
        # Microphone: a capsule, a holder arc, a stem and a base
        c.create_line(cx, cy-26, cx, cy+2, width=26, capstyle=tk.ROUND, fill='white', tags='button')
        c.create_arc(cx-24, cy-26, cx+24, cy+20, start=180, extent=180, style=tk.ARC, width=5, outline='white', tags='button')
        c.create_line(cx, cy+20, cx, cy+32, width=5, fill='white', tags='button')
        c.create_line(cx-13, cy+33, cx+13, cy+33, width=5, capstyle=tk.ROUND, fill='white', tags='button')
        c.tag_lower('circle')
        c.config(cursor='' if self.listening else 'hand2')

    def HoverRecordButton(self, hovering):
        if not self.listening:
            self.canvas.itemconfig('circle', fill=ACCENT_HOVER if hovering else ACCENT)

    def AnimateListening(self): # Pulse rings out from the button while recording
        if not self.listening or not self.canvas.winfo_exists():
            return
        c, cx = self.canvas, self.canvasSize / 2
        c.delete('ring')
        t = timer() - self.start
        for k in range(3):
            phase = ((t / 1.8) + k / 3) % 1           # 0 -> 1 as each ring grows
            r = 84 + phase * 64
            colour = BlendColours(LISTEN, BG, phase)  # fade out as it grows
            c.create_oval(cx-r, cx-r, cx+r, cx+r, outline=colour, width=3, tags='ring')
        c.tag_raise('button')

        elapsed = min(RECORD_SECONDS, int(t))
        self.hintLabel.config(text=f'{elapsed}s / {RECORD_SECONDS}s')
        self.after(33, self.AnimateListening)

    def recordButtonClick(self): # Method to handle record button pressing
        if self.listening:
            return # Already recording - ignore extra clicks
        if self.engine is None:
            self.statusLabel.config(text='Still loading the library…')
            return
        self.listening = True
        self.pointerInAudio = 0

        # Switch to the 'listening' look
        self.DrawRecordButton(LISTEN)
        self.statusLabel.config(text='Listening…')
        self.lastSongsButton.pack_forget()

        # Two background threads run at the same time: one records audio, the other processes that recorded audio in parallel.
        # Background threads never touch the GUI directly (tkinter isn't thread-safe) - they put results on a queue that the GUI checks.
        self.resultQueue = queue.Queue()

        # This thread is begun first and starts recording audio and saving it to a file every second
        self.recordAudioThread = threading.Thread(target=self.RecordAudioSafely, args=(RECORD_SECONDS, os.path.join(os.getcwd(), "RecordedAudio.wav"), 1), daemon=True)
        self.recordAudioThread.start()
        self.start=timer()
        self.AnimateListening()

        # After two seconds (to allow time for some audio to be recorded which can be processed), begin to process the data and look for matches.
        # after() waits without freezing the window, unlike time.sleep()
        self.after(2000, self.StartIdentifying)
        self.after(100, self.CheckForResult)

    def RecordAudioSafely(self, *args): # Run the recording, and report any microphone error to the GUI instead of crashing silently
        try:
            AudioModule.RecordAudio(*args)
        except Exception as e:
            AudioModule.stopCondition = True
            self.resultQueue.put(('error', 'Microphone error: ' + str(e)))

    def StartIdentifying(self):
        # If the recording already stopped (e.g. microphone error), there's nothing to search
        if not self.recordAudioThread.is_alive():
            if self.resultQueue.empty():
                self.resultQueue.put(('none', None))
            return
        self.identifySongThread = threading.Thread(target=self.IDSong, args=(os.path.join(os.getcwd(), "RecordedAudio.wav"),), daemon=True)
        self.identifySongThread.start()

    def SecondsPassed(self): # Seconds since recording started, capped at 10 (used to scale the match threshold)
        return min(10, int(timer() - self.start))

    def CheckForResult(self): # Runs on the GUI thread: show the result once the background search has finished
        try:
            kind, payload = self.resultQueue.get_nowait()
        except queue.Empty:
            self.after(100, self.CheckForResult)
            return

        self.listening = False
        searchTime = str(round(timer() - self.start, 1))

        if kind == 'match':
            song = payload['song']
            self.lastSongsMatched.append(song['title'] + ' by ' + song['artist'])
            details = (f"Heard at {EngineModule.format_time(payload['offset_sec'])} in the song   ·   "
                       f"chance of a false match < {FormatP(payload['p_value'])}")
            self.ShowResults(song['title'], song['artist'], song.get('album', ''), song.get('year', ''), searchTime, details=details)
        elif kind == 'error':
            print(payload)
            self.ShowResults(0,0,0,0,searchTime, message = payload)
        else:
            self.ShowResults(0,0,0,0,searchTime)

    # ---------- Result screen ----------
    def ShowResults(self, name, artist, album, releaseDate, searchTime, message = '', details = ''):
        screen = self.NewScreen()

        card = tk.Frame(screen, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=(40, 0), ipady=22)

        if name == 0:
            # Nothing found (or an error happened)
            tk.Label(card, text='?', font=self.fonts['icon'], bg=SURFACE, fg=WARNING).pack(pady=(14, 0))
            tk.Label(card, text='No match found', font=self.fonts['songTitle'], bg=SURFACE, fg=TEXT).pack(pady=(4, 6))
            tk.Label(card, text=message or "Try getting closer to the speaker,\nor make sure the song is in the library.",
                     font=self.fonts['meta'], bg=SURFACE, fg=MUTED, wraplength=300, justify=tk.CENTER).pack(padx=20)
        else:
            tk.Label(card, text='MATCH FOUND', font=self.fonts['eyebrow'], bg=SURFACE, fg=SUCCESS).pack(pady=(14, 8))
            # Long titles wrap onto more lines instead of being cut off
            tk.Label(card, text=str(name), font=self.fonts['songTitle'], bg=SURFACE, fg=TEXT,
                     wraplength=310, justify=tk.CENTER).pack(padx=20)
            tk.Label(card, text=str(artist), font=self.fonts['artist'], bg=SURFACE, fg=ACCENT).pack(pady=(6, 14))

            # Album / single and release year shown as one small line
            albumText = 'Single' if album == 'Single' else str(album)
            tk.Frame(card, bg=BORDER, height=1).pack(fill=tk.X, padx=28, pady=(0, 12))
            meta = '   ·   '.join(x for x in (albumText, str(releaseDate)) if x and x != 'nan')
            tk.Label(card, text=meta, font=self.fonts['meta'], bg=SURFACE, fg=MUTED, wraplength=310).pack()
            if details:
                tk.Label(card, text=details, font=self.fonts['small'], bg=SURFACE, fg=MUTED, wraplength=320,
                         justify=tk.CENTER).pack(pady=(8, 0))

            # Save the stack of last matched songs to the file
            self.SaveHistory()

        tk.Label(screen, text=('Found in ' if name != 0 else 'Searched for ') + searchTime + 's',
                 font=self.fonts['small'], bg=BG, fg=MUTED).pack(pady=(12, 0))

        # Buttons
        buttons = tk.Frame(screen, bg=BG)
        buttons.pack(side=tk.BOTTOM, pady=(0, 6))
        FlatButton(buttons, 'Listen again', self.ListenAgain, font=self.fonts['button']).pack(side=tk.LEFT, padx=6)
        self.backButton = FlatButton(buttons, 'Home', self.ResetWidgets, bg=SURFACE, hover=SURFACE_HOVER, font=self.fonts['button'])
        self.backButton.pack(side=tk.LEFT, padx=6)

    def ListenAgain(self): # Go straight back to listening
        self.ShowHome()
        self.recordButtonClick()

    # ---------- History screen ----------
    def ShowLastMatches(self):
        screen = self.NewScreen()

        top = tk.Frame(screen, bg=BG)
        top.pack(fill=tk.X, pady=(16, 12))
        self.backButton = FlatButton(top, '←  Back', self.ResetWidgets, bg=BG, hover=SURFACE, fg=MUTED,
                                     font=self.fonts['button'], padx=10, pady=4)
        self.backButton.pack(side=tk.LEFT)
        tk.Label(top, text='Recent matches', font=self.fonts['heading'], bg=BG, fg=TEXT).pack(side=tk.LEFT, padx=(10, 0))

        self.historyList = tk.Frame(screen, bg=BG)
        self.historyList.pack(fill=tk.BOTH, expand=True)
        self.arrayOfLabels = []

        if len(self.lastSongsMatched) == 0:
            self.ShowEmptyHistory()
            return

        # Show the most recent match first, without removing anything from the list
        for i, match in enumerate(reversed(self.lastSongsMatched)):
            row = tk.Frame(self.historyList, bg=SURFACE)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=str(i+1), font=self.fonts['row'], bg=SURFACE, fg=MUTED, width=3).pack(side=tk.LEFT, pady=8)
            currentLabel = tk.Label(row, text=match, font=self.fonts['row'], bg=SURFACE, fg=TEXT, anchor=tk.W,
                                    justify=tk.LEFT, wraplength=280)
            currentLabel.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8, padx=(0, 10))
            self.arrayOfLabels.append(currentLabel)

        self.clearButton = FlatButton(screen, 'Clear history', self.ClearPreviousMatches, bg=SURFACE, hover='#3a2230',
                                      fg=LISTEN, font=self.fonts['button'])
        self.clearButton.pack(side=tk.BOTTOM, pady=(10, 6))

    def ShowEmptyHistory(self):
        tk.Label(self.historyList, text='No matches yet', font=self.fonts['status'], bg=BG, fg=TEXT).pack(pady=(80, 4))
        tk.Label(self.historyList, text='Songs you identify will show up here.', font=self.fonts['subtitle'],
                 bg=BG, fg=MUTED).pack()

    def ClearPreviousMatches(self):
        self.lastSongsMatched.clear()
        self.SaveHistory()
        for child in self.historyList.winfo_children():
            child.destroy()
        self.arrayOfLabels = []
        self.clearButton.destroy()
        self.ShowEmptyHistory()


    def IDSong(self, filepath): # Runs on a background thread: re-check the growing recording every second
        # Note: this never touches the GUI - the result goes on the queue and CheckForResult (on the GUI thread) displays it.
        # Every check uses the statistical decision of EngineModule; because we look up to RECORD_SECONDS times,
        # each look gets alpha / RECORD_SECONDS of the error budget (overall false-match rate stays below alpha).
        resultQueue = self.resultQueue
        recordThread = self.recordAudioThread
        lastFrames = 0
        try:
            while True:
                # Check this before reading, so the last bit of audio still gets checked after the recording ends
                recordingFinished = not recordThread.is_alive()
                try:
                    with wave.open(filepath, 'rb') as w:
                        frames, sr = w.getnframes(), w.getframerate()
                        audio = np.frombuffer(w.readframes(frames), dtype='<i2').astype(np.float64) if frames > lastFrames else None
                except (wave.Error, EOFError, FileNotFoundError):
                    audio = None
                if audio is None:
                    if recordingFinished:
                        resultQueue.put(('none', None))
                        return
                    time.sleep(0.2)
                    continue
                lastFrames = frames
                result = self.engine.identify(audio, sr, looks=RECORD_SECONDS)
                print(f"{frames / sr:4.1f}s heard: best score {result['score']}, p = {result['p_value']:.2e}")
                if result['matched']:
                    AudioModule.stopCondition = True
                    resultQueue.put(('match', result))
                    return
                if recordingFinished:
                    resultQueue.put(('none', None))
                    return
        except Exception as e:
            AudioModule.stopCondition = True
            resultQueue.put(('error', 'Error while searching: ' + str(e)))


SUPERSCRIPT = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def FormatP(p): # Short human-readable probability, e.g. 1 in 10¹²
    e = 15 if (p <= 0 or p < 1e-15) else int(np.floor(-np.log10(p)))
    return f'1 in {10 ** e:,}' if e <= 6 else '1 in 10' + str(e).translate(SUPERSCRIPT)
