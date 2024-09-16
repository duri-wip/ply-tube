# def preprocessing(playlists):
#         cleaned_playlists = []
#         for playlist in playlists:
#             cleaned_playlist = []
#             for item in playlist['items']:
#                 item = item.replace(u'\xa0', u' ')
#                 item = item.strip()
#                 if ' - ' in item:
#                     if len(item.split(' - ')) == 2:
#                         cleaned_playlist.append(item)
#                 elif '-' in item:
#                     if len(item.split('-')) == 2:
#                         cleaned_playlist.append(item.replace('-', ' - '))
#             cleaned_playlists.append(cleaned_playlist)
#         return cleaned_playlists

def preprocessing(playlists):
    if not isinstance(playlists, list):
        raise TypeError(f"Expected playlists to be a list, but got {type(playlists)}")
    
    cleaned_playlists = []
    for playlist in playlists:
        cleaned_playlist = []
        for item in playlist.get('items', []):  # 'items'가 없는 경우 기본 빈 리스트
            item = item.replace(u'\xa0', u' ').strip()

            if ' - ' in item or '-' in item:
                parts = item.replace('-', ' - ').split(' - ')
                if len(parts) == 2:
                    cleaned_playlist.append(' - '.join(parts))
        cleaned_playlists.append(cleaned_playlist)
    return cleaned_playlists
