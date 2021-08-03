import lyricsgenius
import pandas as pd
import numpy as np
from typing import List
from functools import wraps
import datetime as dt

def log_step(f, *args, **kwargs):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = dt.datetime.now()
        result = f(*args, **kwargs)
        te = dt.datetime.now()
        print(f"{f.__name__}({args},{kwargs}) took: {te - ts} sec")
        return result
    return wrap

@log_step
def download_lyrics(access_token: str, artist: str, fname: str=None, exclude: List=['(Remix)', '(Live)']) -> None:
    genius = lyricsgenius.Genius()
    genius.remove_section_headers = True
    genius.excluded_terms = exclude
    artist_ = genius.search_artist(artist)
    artist_.save_lyrics(fname)

@log_step
def load_lyrics(data: dict) -> pd.DataFrame:
    songs_dict = {
        'title': [],
        'artist': [],
        'release_date': [],
        'album': [],
        'lyrics': []
    }

    for i in range(len(data['songs'])):
        songs_dict['title'].append(data['songs'][i]['title'])
        songs_dict['artist'].append(data['songs'][i]['artist'])
        # TODO: Fix if album is N/A
        if data['songs'][i]['album'] is not None:
            songs_dict['album'].append(data['songs'][i]['album']['name'])
        else:
            songs_dict['album'].append(None)
        songs_dict['release_date'].append(data['songs'][i]['release_date'])
        songs_dict['lyrics'].append(data['songs'][i]['lyrics'])
        
    return pd.DataFrame(songs_dict)

@log_step
def load_embeddings(fname: str) -> dict:
    glove_embeddings = {}
    with open(fname) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], "float32")
            glove_embeddings[word] = coefs
    f.close()

    print(f"Found {len(glove_embeddings)} word vectors.")
    
    return glove_embeddings