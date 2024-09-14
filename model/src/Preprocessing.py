def preprocessing(playlists):
        cleaned_playlists = []
        for playlist in playlists:
            cleaned_playlist = []
            for item in playlist['items']:
                item = item.replace(u'\xa0', u' ')
                item = item.strip()
                if ' - ' in item:
                    if len(item.split(' - ')) == 2:
                        cleaned_playlist.append(item)
                elif '-' in item:
                    if len(item.split('-')) == 2:
                        cleaned_playlist.append(item.replace('-', ' - '))
            cleaned_playlists.append(cleaned_playlist)
        return cleaned_playlists
