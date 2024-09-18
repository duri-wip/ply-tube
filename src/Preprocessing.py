import re

def remove_emoticons(text):
    # 이모티콘을 감지하기 위한 정규 표현식
    emoticon_pattern = re.compile(
        r'[\U0001F600-\U0001F64F'  # 이모티콘: 표정
        r'\U0001F300-\U0001F5FF'  # 이모티콘: 기호 및 물건
        r'\U0001F680-\U0001F6FF'  # 이모티콘: 교통 및 지도
        r'\U0001F700-\U0001F77F'  # 이모티콘: 기호
        r'\U0001F780-\U0001F7FF'  # 이모티콘: 기호
        r'\U0001F800-\U0001F8FF'  # 이모티콘: 기호
        r'\U0001F900-\U0001F9FF'  # 이모티콘: 기호
        r'\U0001FA00-\U0001FA6F'  # 이모티콘: 기호
        r'\U0001FA70-\U0001FAFF'  # 이모티콘: 기호
        r'\U00002702-\U000027B0'  # 이모티콘: 다양한 기호
        r'\U000024C2-\U0001F251'  # 이모티콘: 추가 기호
        r']', re.UNICODE)
    
    return emoticon_pattern.sub('', text)

def preprocessing(playlists):
        cleaned_playlists = []
        for playlist in playlists:
            cleaned_playlist = []
            for item in playlist['items']:
                item = item.replace(u'\xa0', u' ')
                item = item.strip()
                item = remove_emoticons(item)
                if ' - ' in item:
                    if len(item.split(' - ')) == 2:
                        cleaned_playlist.append(item)
                elif '-' in item:
                    if len(item.split('-')) == 2:
                        cleaned_playlist.append(item.replace('-', ' - '))
            cleaned_playlists.append(cleaned_playlist)
        return cleaned_playlists

