"""Add a song to the MusicID library, or identify a recording, from the command line.

python add_song.py            (asks questions)
Any WAV/FLAC/OGG/MP3 file at any sample rate works (it is converted to mono 11 kHz internally).
"""
import soundfile as sf
import EngineModule


def load(path):
    audio, sr = sf.read(path, dtype='float32')
    return audio, sr


def add_song_to_database(song_path, song_name, artist, album, release_date):
    """Adds a song's fingerprint to the library (refuses exact duplicates of title + artist)."""
    engine = EngineModule.Engine()
    if any(s['title'].lower() == song_name.lower() and s['artist'].lower() == artist.lower() for s in engine.songs):
        print(f"'{song_name}' by {artist} is already in the library.")
        return None
    audio, sr = load(song_path)
    sid = engine.add_song(audio, sr, {'title': song_name, 'artist': artist, 'album': album, 'year': release_date})
    engine.save()
    print(f"Song '{song_name}' added to the library (id {sid}, {engine.n_songs:,} songs in total).")
    return sid


def detect_song_from_database(audio_path):
    """Identifies a recording and prints the result with its statistical confidence."""
    engine = EngineModule.Engine()
    audio, sr = load(audio_path)
    r = engine.identify(audio, sr)
    if r['matched']:
        s = r['song']
        print(f"Detected Song: {s['title']} by {s['artist']}")
        if s.get('album') and s['album'] != 'Single':
            print(f"Album: {s['album']}")
        if s.get('year'):
            print(f"Release Date: {s['year']}")
        print(f"Heard at {EngineModule.format_time(r['offset_sec'])} in the song; score {r['score']}, "
              f"family-wise false-match probability p = {r['p_value']:.1e}")
    else:
        print(f"No matching song found (best score {r['score']}, p = {r['p_value']:.2f}).")
    return r


if __name__ == "__main__":
    mode = input("Enter 'add' to add a song or 'detect' to identify a song: ").strip().lower()
    if mode == 'add':
        song_path = input("Enter the path to the song (audio file): ").strip()
        song_name = input("Enter song name: ").strip()
        artist = input("Enter artist name: ").strip()
        album = input("Enter album name (or 'Single' if none): ").strip()
        release_date = input("Enter release date: ").strip()
        add_song_to_database(song_path, song_name, artist, album, release_date)
    elif mode == 'detect':
        audio_path = input("Enter the path to the recorded audio: ").strip()
        detect_song_from_database(audio_path)
    else:
        print("Invalid input. Use 'add' or 'detect'.")
