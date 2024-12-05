import os
import sys
import uuid
import jwt
import time
import secrets
import datetime
from typing import List, Optional, Dict, Any

# Explicitly handle spotipy import
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError as e:
    print(f"Import Error: {e}")
    spotipy = None

from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    Request, 
    status
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Logging for debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyCredentialsError(Exception):
    """Custom exception for Spotify credentials issues"""
    pass

# Simplified Token Management for Serverless
class InMemoryTokenManager:
    def __init__(self):
        self.tokens = {}
        self.request_counts = {}
    
    def generate_secure_token(self, ip_address: str = None, user_agent: str = None) -> str:
        token_id = str(uuid.uuid4())
        
        payload = {
            'jti': token_id,
            'iat': datetime.datetime.now(datetime.timezone.utc),
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
            'ip': ip_address,
            'user_agent': user_agent
        }
        
        secret_key = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
        
        encoded_token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        self.tokens[encoded_token] = {
            'created_at': datetime.datetime.now(datetime.timezone.utc),
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        return encoded_token
    
    def validate_token(self, token: str, endpoint: str, ip_address: str) -> bool:
        try:
            secret_key = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
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
        # Explicit error handling for Spotify credentials
        client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')

        logger.info(f"Initializing Spotify Service with Client ID: {client_id}")

        if not client_id or not client_secret:
            logger.error("Spotify credentials not found in environment")
            raise SpotifyCredentialsError("Spotify credentials not found in environment")
        
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise

    def search_tracks(
        self, 
        query: str, 
        search_type: str = 'track', 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Searching for {query} with type {search_type}")
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
                        'tracks': playlist['tracks']['total'],
                        'owner': playlist['owner']['display_name']
                    }
                    for playlist in results['playlists']['items']
                ]
        except Exception as e:
            logger.error(f"Search error: {e}")
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
            logger.info(f"Getting top tracks for artist: {artist_name}")
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
            logger.error(f"Top tracks error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Top tracks error: {str(e)}"
            )

# Pydantic Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    search_type: Optional[str] = Field(default='track', regex='^(track|album|artist|playlist)$')
    limit: Optional[int] = Field(default=10, ge=1, le=50)

class TopTracksRequest(BaseModel):
    artist_name: str = Field(..., min_length=1, max_length=100)
    limit: Optional[int] = Field(default=10, ge=1, le=50)

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
token_manager = InMemoryTokenManager()

# Attempt to initialize Spotify Service with error handling
try:
    spotify_service = SpotifyTrackService()
except SpotifyCredentialsError:
    logger.error("Failed to initialize Spotify Service")
    spotify_service = None

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
@app.get("/")
async def root():
    return {
        "message": "Spotify Track Retrieval API is running",
        "spotify_service_status": "Initialized" if spotify_service else "Not Initialized"
    }

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
    if not spotify_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify service not initialized"
        )
    
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
    if not spotify_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify service not initialized"
        )
    
    client_ip = request.client.host
    token_manager.validate_token(token, '/top-tracks', client_ip)
    
    return spotify_service.get_artist_top_tracks(
        top_tracks_req.artist_name, 
        top_tracks_req.limit
    )

# Global Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }

# Vercel Handler (optional)
def handler(event, context):
    return app
