def preprocess(item):
    item = item.lower()
    
    item_split = item.split('-')
    if len(item_split) == 2:
        item_split = list(map(lambda x: x.replace(' ',''), item_split))
        item = ' - '.join(item_split)
    return item

def preprocessing(playlists):
    cleaned_playlists = []
    for playlist in playlists['playlists']:
        cleaned_playlist = []
        for item in playlist['items']:
            if preprocess(item) is not None:
                cleaned_playlist.append(preprocess(item))
        if not cleaned_playlist:
            pass
        else:
            cleaned_playlists.append(list(set(cleaned_playlist)))

    print("-----------데이터 전처리 완료-----------")
    return cleaned_playlists


