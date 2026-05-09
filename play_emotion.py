import spotipy
from spotipy.oauth2 import SpotifyOAuth

# 1. Kết nối API Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="f66ced5a6d7c4b9fac274fa25a7df9d8",
    client_secret="a1fd7b07e77b455aa232e001a18f724d",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-playback-state user-modify-playback-state user-read-currently-playing app-remote-control streaming",
    cache_path=".cache-emotion"
))


# 2. Ánh xạ cảm xúc -> Playlist Spotify
emotion_to_playlist = {
    "happy": "spotify:playlist:0q5iPk7eypDc3ThgjeL6K7?si=bac6c9ddeb9d4cac",
    "sad":   "spotify:playlist:0q5iPk7eypDc3ThgjeL6K7?si=bac6c9ddeb9d4cac",
    "angry": "spotify:playlist:37i9dQZF1DWYxwmBaMqxsl",
    "neutral": "spotify:playlist:1xSl5sRm8JJRj7Gwh6gf9R?si=a4a78a8bdbd947d9&pt=d6baf66b805c1046a5acbd785f920c6b"
}

# 3. Hàm phát playlist
def play_playlist(emotion):
    playlist_uri = emotion_to_playlist.get(emotion)
    if playlist_uri:
        try:
            sp.start_playback(context_uri=playlist_uri)
            print(f"🎶 Đang phát playlist cho cảm xúc: {emotion}")
        except Exception as e:
            print("Lỗi khi phát nhạc:", e)
    else:
        print("❌ Không tìm thấy playlist cho cảm xúc:", emotion)


# --- Test ---
if __name__ == "__main__":
    play_playlist("happy")   # giả lập emotion = happy
