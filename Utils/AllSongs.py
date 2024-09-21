import json
from src.Preprocessing import preprocessing
from sklearn.preprocessing import LabelEncoder


def generate_all_songs():
    with open("/home/ubuntu/feature_store/playlist.json", 'r') as f:
        playlists = json.load(f)

    playlists = preprocessing(playlists)

    all_songs = [song for playlist in playlists for song in playlist]

    # Fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(all_songs)

    return label_encoder

