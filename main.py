import os
import uuid
import jwt
import time
import sqlite3
import secrets
import datetime
from typing import List, Optional, Dict, Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    Request, 
    status
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Token Security Manager
class SecureTokenManager:
    def __init__(self, database_path='token_management.db'):
        self.conn = sqlite3.connect(database_path, check_same_thread=False)
        self.create_tables()
        self.request_counts = {}
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                last_used TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_logs (
                token TEXT,
                timestamp TIMESTAMP,
                endpoint TEXT,
                ip_address TEXT
            )
        ''')
        self.conn.commit()
    
    def generate_secure_token(self, ip_address: str = None, user_agent: str = None) -> str:
        token_id = str(uuid.uuid4())
        
        payload = {
            'jti': token_id,
            'iat': datetime.datetime.now(datetime.timezone.utc),
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
            'ip': ip_address,
            'user_agent': user_agent
        }
        
        secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
        
        encoded_token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO tokens 
            (token, created_at, last_used, ip_address, user_agent) 
            VALUES (?, ?, ?, ?, ?)
        ''', (
            encoded_token, 
            datetime.datetime.now(datetime.timezone.utc), 
            datetime.datetime.now(datetime.timezone.utc),
            ip_address,
            user_agent
        ))
        self.conn.commit()
        
        return encoded_token
    
    def validate_token(self, token: str, endpoint: str, ip_address: str) -> bool:
        try:
            secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
            decoded_token = jwt.decode(
                token, 
                secret_key, 
                algorithms=['HS256']
            )
            
            current_time = time.time()
            
            # Rate limiting
            self.request_counts[token] = [
                req_time for req_time in self.request_counts.get(token, []) 
                if current_time - req_time < 60
            ]
            
            if len(self.request_counts.get(token, [])) >= 100:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                    detail="Rate limit exceeded"
                )
            
            self.request_counts.setdefault(token, []).append(current_time)
            
            # Log request
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO request_logs 
                (token, timestamp, endpoint, ip_address) 
                VALUES (?, ?, ?, ?)
            ''', (
                token, 
                datetime.datetime.now(datetime.timezone.utc), 
                endpoint, 
                ip_address
            ))
            self.conn.commit()
            
            # Update last used timestamp
            cursor.execute('''
                UPDATE tokens 
                SET last_used = ? 
                WHERE token = ?
            ''', (
                datetime.datetime.now(datetime.timezone.utc),
                token
            ))
            self.conn.commit()
            
            return True
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )

# Spotify Track Service
class SpotifyTrackService:
    def __init__(self):
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

        if not client_id or not client_secret:
            raise ValueError("Spotify credentials not found in environment")
        
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, 
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def search_tracks(
        self, 
        query: str, 
        search_type: str = 'track', 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            if query.startswith('https://open.spotify.com/'):
                # Link handling (same as before)
                if 'playlist' in query:
                    playlist_id = query.split('/')[-1].split('?')[0]
                    playlist = self.sp.playlist(playlist_id)
                    return [
                        {
                            'name': track['track']['name'],
                            'artist': track['track']['artists'][0]['name'],
                            'album': track['track']['album']['name'],
                            'id': track['track']['id']
                        }
                        for track in playlist['tracks']['items']
                    ]
                elif 'album' in query:
                    album_id = query.split('/')[-1].split('?')[0]
                    album = self.sp.album(album_id)
                    return [
                        {
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'album': album['name'],
                            'id': track['id']
                        }
                        for track in album['tracks']['items']
                    ]
                else:
                    track_id = query.split('/')[-1].split('?')[0]
                    track = self.sp.track(track_id)
                    return [
                        {
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'album': track['album']['name'],
                            'id': track['id']
                        }
                    ]
            else:
                results = self.sp.search(q=query, type=search_type, limit=limit)
                
                if search_type == 'track':
                    return [
                        {
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'album': track['album']['name'],
                            'id': track['id']
                        }
                        for track in results['tracks']['items']
                    ]
                elif search_type == 'album':
                    return [
                        {
                            'name': album['name'],
                            'artist': album['artists'][0]['name'],
                            'id': album['id']
                        }
                        for album in results['albums']['items']
                    ]
                elif search_type == 'artist':
                    return [
                        {
                            'name': artist['name'],
                            'id': artist['id'],
                            'genres': artist.get('genres', [])
                        }
                        for artist in results['artists']['items']
                    ]
                elif search_type == 'playlist':
                    return [
                        {
                            'name': playlist['name'],
                            'id': playlist['id'],
                            'tracks': len(playlist['tracks']['items']),
                            'owner': playlist['owner']['display_name']
                        }
                        for playlist in results['playlists']['items']
                    ]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Search error: {str(e)}"
            )

    def get_artist_top_tracks(
        self, 
        artist_name: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            artist_results = self.sp.search(q=artist_name, type='artist', limit=1)
            
            if not artist_results['artists']['items']:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Artist not found"
                )
            
            artist_id = artist_results['artists']['items'][0]['id']
            top_tracks = self.sp.artist_top_tracks(artist_id)
            
            return [
                {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'id': track['id']
                }
                for track in top_tracks['tracks'][:limit]
            ]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Top tracks error: {str(e)}"
            )

# Pydantic Models
class SearchRequest(BaseModel):
    query: str
    search_type: Optional[str] = 'track'
    limit: Optional[int] = 10

class TopTracksRequest(BaseModel):
    artist_name: str
    limit: Optional[int] = 10

# FastAPI Application
app = FastAPI(
    title="Secure Spotify Track Retrieval API",
    description="A secure API for retrieving Spotify tracks, albums, playlists, and artist information"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Managers
token_manager = SecureTokenManager()
spotify_service = SpotifyTrackService()

# Authorization Dependency
security = HTTPBearer()

def validate_request(
    request: Request, 
    token: HTTPAuthorizationCredentials = Depends(security)
):
    client_ip = request.client.host
    token_manager.validate_token(token.credentials, request.url.path, client_ip)
    return token.credentials

# Routes
@app.post("/token")
async def generate_token(request: Request):
    client_ip = request.client.host
    user_agent = request.headers.get('User-Agent', 'Unknown')
    return {
        "token": token_manager.generate_secure_token(client_ip, user_agent)
    }

@app.post("/search")
async def search_tracks(
    search_req: SearchRequest,
    token: str = Depends(validate_request),
    request: Request = None
):
    client_ip = request.client.host
    token_manager.validate_token(token, '/search', client_ip)
    
    return spotify_service.search_tracks(
        search_req.query, 
        search_req.search_type, 
        search_req.limit
    )

@app.post("/top-tracks")
async def get_top_tracks(
    top_tracks_req: TopTracksRequest,
    token: str = Depends(validate_request),
    request: Request = None
):
    client_ip = request.client.host
    token_manager.validate_token(token, '/top-tracks', client_ip)
    
    return spotify_service.get_artist_top_tracks(
        top_tracks_req.artist_name, 
        top_tracks_req.limit
    )

# Global Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }

# Main Entry Point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
