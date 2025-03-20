import os
import sys
import uuid
import jwt
import time
import secrets
import datetime
import re
from typing import List, Optional, Dict, Any, Union
import asyncio
import aiohttp
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Spotipy import with proper error handling
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError as e:
    logger.error(f"Spotipy import error: {e}")
    spotipy = None

# Cache management
class SpotifyCache:
    def __init__(self, max_size=1000, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        # Check if cached item has expired
        if time.time() - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key, value):
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# Token management
class TokenManager:
    def __init__(self):
        self.tokens = {}
        self.request_counts = {}
        self.secret_key = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
    
    def generate_token(self, ip_address: str = None, user_agent: str = None) -> str:
        """Generate a secure JWT token for API access"""
        token_id = str(uuid.uuid4())
        
        payload = {
            'jti': token_id,
            'iat': datetime.datetime.now(datetime.timezone.utc),
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24),  # Longer expiry
            'ip': ip_address,
            'ua': user_agent[:100] if user_agent else None  # Truncate long user agents
        }
        
        encoded_token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        self.tokens[encoded_token] = {
            'created_at': datetime.datetime.now(datetime.timezone.utc),
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        return encoded_token
    
    def validate_token(self, token: str, endpoint: str = None, ip_address: str = None) -> bool:
        """Validate a token and handle rate limiting"""
        try:
            # Decode and verify the token
            decoded_token = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Rate limiting - track requests per minute
            current_time = time.time()
            token_key = decoded_token['jti']
            
            # Keep only requests from the last minute
            self.request_counts[token_key] = [
                req_time for req_time in self.request_counts.get(token_key, []) 
                if current_time - req_time < 60
            ]
            
            # Check rate limit (120 requests per minute)
            if len(self.request_counts.get(token_key, [])) >= 120:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                    detail="Rate limit exceeded: 120 requests per minute"
                )
            
            # Record this request
            self.request_counts.setdefault(token_key, []).append(current_time)
            
            return True
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )

# Models
class SpotifyTrack(BaseModel):
    """Model for a Spotify track with essential metadata"""
    id: str
    name: str
    artist: str
    artist_id: str
    album: str
    album_id: str
    duration_ms: int
    explicit: bool
    popularity: int
    preview_url: Optional[str] = None
    image_url: Optional[str] = None
    release_date: Optional[str] = None
    
    @validator('image_url', pre=True, always=True)
    def set_default_image(cls, v):
        return v or "https://i.scdn.co/image/ab67616d0000b273000000000000000000000000"

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    search_type: str = Field(default='track', pattern=r'^(track|album|artist|playlist|url)$')
    limit: int = Field(default=20, ge=1, le=50)
    
    @validator('query')
    def sanitize_query(cls, v):
        # Basic sanitization
        return v.strip()

class TopTracksRequest(BaseModel):
    artist_id: Optional[str] = None
    artist_name: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    
    @validator('artist_name', 'artist_id', always=True)
    def check_artist_info(cls, v, values):
        # Ensure at least one of artist_id or artist_name is provided
        if not values.get('artist_id') and not values.get('artist_name') and not v:
            raise ValueError("Either artist_id or artist_name must be provided")
        return v

class DownloadRequest(BaseModel):
    track_id: str
    name: str
    artist: str

# Spotify Service
class SpotifyService:
    def __init__(self):
        client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            logger.error("Spotify credentials not found in environment variables")
            raise ValueError("Missing Spotify credentials")
            
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(
                client_credentials_manager=client_credentials_manager,
                requests_timeout=10,
                retries=3
            )
            self.cache = SpotifyCache()
            logger.info("Spotify service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise
    
    def _format_track(self, track: Dict) -> SpotifyTrack:
        """Format a raw Spotify track object into our model"""
        if not track or not isinstance(track, dict):
            return None
            
        # Extract the largest available image
        images = track.get('album', {}).get('images', [])
        image_url = None
        if images:
            # Sort by size (width) descending
            sorted_images = sorted(images, key=lambda x: x.get('width', 0), reverse=True)
            image_url = sorted_images[0].get('url') if sorted_images else None
        
        return SpotifyTrack(
            id=track.get('id', ''),
            name=track.get('name', 'Unknown Track'),
            artist=track.get('artists', [{}])[0].get('name', 'Unknown Artist'),
            artist_id=track.get('artists', [{}])[0].get('id', ''),
            album=track.get('album', {}).get('name', 'Unknown Album'),
            album_id=track.get('album', {}).get('id', ''),
            duration_ms=track.get('duration_ms', 0),
            explicit=track.get('explicit', False),
            popularity=track.get('popularity', 0),
            preview_url=track.get('preview_url'),
            image_url=image_url,
            release_date=track.get('album', {}).get('release_date')
        )
    
    def _extract_spotify_details(self, url: str) -> Optional[Dict[str, str]]:
        """Extract Spotify entity type and ID from a URL"""
        # Standard Spotify URL pattern
        patterns = [
            r'spotify\.com/(track|album|playlist|artist)/([a-zA-Z0-9]+)',
            r'spotify:([a-z]+):([a-zA-Z0-9]+)'  # URI format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    'type': match.group(1),
                    'id': match.group(2)
                }
        return None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse the query to determine if it's a URL or text search"""
        # Check if query is a Spotify URL
        spotify_details = self._extract_spotify_details(query)
        if spotify_details:
            return {
                'type': 'url',
                'entity_type': spotify_details['type'],
                'entity_id': spotify_details['id']
            }
            
        # Check if it's a YouTube URL
        youtube_patterns = [
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, query)
            if match:
                playlist_match = re.search(r'list=([a-zA-Z0-9_-]+)', query)
                return {
                    'type': 'youtube',
                    'entity_type': 'playlist' if playlist_match else 'video',
                    'entity_id': playlist_match.group(1) if playlist_match else match.group(1)
                }
                
        # Regular text search
        return {
            'type': 'text',
            'query': query
        }
    
    async def search_tracks(self, query: str, search_type: str = 'track', limit: int = 20) -> List[SpotifyTrack]:
        """Search for tracks based on query and type"""
        cache_key = f"search:{query}:{search_type}:{limit}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Parse the query
            parsed = self.parse_query(query)
            
            # If it's a URL, handle it differently
            if parsed['type'] == 'url':
                entity_type = parsed['entity_type']
                entity_id = parsed['entity_id']
                
                if entity_type == 'track':
                    track = self.sp.track(entity_id)
                    result = [self._format_track(track)]
                    
                elif entity_type == 'album':
                    album_tracks = self.sp.album_tracks(entity_id, limit=limit)
                    album = self.sp.album(entity_id)
                    
                    result = []
                    for item in album_tracks['items']:
                        # Add album details to each track
                        item['album'] = {
                            'name': album['name'],
                            'id': album['id'],
                            'images': album['images'],
                            'release_date': album['release_date']
                        }
                        result.append(self._format_track(item))
                    
                elif entity_type == 'playlist':
                    tracks = []
                    playlist = self.sp.playlist(entity_id)
                    
                    # Get initial tracks
                    results = self.sp.playlist_items(
                        entity_id, 
                        limit=min(limit, 100),
                        fields='items(track(id,name,artists,album,duration_ms,explicit,popularity,preview_url))'
                    )
                    
                    for item in results['items']:
                        if item.get('track'):
                            tracks.append(self._format_track(item['track']))
                    
                    result = tracks
                
                elif entity_type == 'artist':
                    artist = self.sp.artist(entity_id)
                    top_tracks = self.sp.artist_top_tracks(entity_id)
                    
                    result = [self._format_track(track) for track in top_tracks['tracks'][:limit]]
                    
                else:
                    # Fallback to regular search
                    results = self.sp.search(q=query, type='track', limit=limit)
                    result = [self._format_track(track) for track in results['tracks']['items']]
            
            # YouTube URL - extract info and search on Spotify
            elif parsed['type'] == 'youtube':
                # For YouTube, we need to extract title and search on Spotify
                # This would be handled by a YouTube info extractor
                # For now, just search the URL text
                results = self.sp.search(q=query, type='track', limit=limit)
                result = [self._format_track(track) for track in results['tracks']['items']]
            
            # Regular text search
            else:
                # If search_type is specified, use it
                if search_type == 'track':
                    results = self.sp.search(q=query, type='track', limit=limit)
                    result = [self._format_track(track) for track in results['tracks']['items']]
                    
                elif search_type == 'album':
                    results = self.sp.search(q=query, type='album', limit=limit)
                    
                    # For each album, get its tracks
                    result = []
                    for album in results['albums']['items'][:min(5, limit)]:  # Limit to avoid too many API calls
                        album_tracks = self.sp.album_tracks(album['id'])
                        
                        for track in album_tracks['items']:
                            # Add album details to each track
                            track['album'] = {
                                'name': album['name'],
                                'id': album['id'],
                                'images': album['images'],
                                'release_date': album.get('release_date')
                            }
                            result.append(self._format_track(track))
                        
                        if len(result) >= limit:
                            break
                    
                elif search_type == 'artist':
                    results = self.sp.search(q=query, type='artist', limit=min(5, limit))
                    
                    # For each artist, get top tracks
                    result = []
                    for artist in results['artists']['items']:
                        top_tracks = self.sp.artist_top_tracks(artist['id'])
                        
                        for track in top_tracks['tracks']:
                            result.append(self._format_track(track))
                            
                        if len(result) >= limit:
                            break
                    
                elif search_type == 'playlist':
                    results = self.sp.search(q=query, type='playlist', limit=min(3, limit))
                    
                    # For each playlist, get its tracks
                    result = []
                    for playlist in results['playlists']['items']:
                        playlist_tracks = self.sp.playlist_items(
                            playlist['id'], 
                            limit=min(limit, 50),
                            fields='items(track(id,name,artists,album,duration_ms,explicit,popularity,preview_url))'
                        )
                        
                        for item in playlist_tracks['items']:
                            if item.get('track'):
                                result.append(self._format_track(item['track']))
                                
                        if len(result) >= limit:
                            break
                
                else:
                    # Default to track search
                    results = self.sp.search(q=query, type='track', limit=limit)
                    result = [self._format_track(track) for track in results['tracks']['items']]
            
            # Filter out None values and limit results
            result = [r for r in result if r is not None][:limit]
            
            # Cache the result
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Search error: {str(e)}"
            )
    
    async def get_artist_top_tracks(self, artist_id: Optional[str] = None, artist_name: Optional[str] = None, limit: int = 10) -> List[SpotifyTrack]:
        """Get top tracks for an artist by ID or name"""
        if not artist_id and not artist_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either artist_id or artist_name must be provided"
            )
            
        cache_key = f"artist:{'id:'+artist_id if artist_id else 'name:'+artist_name}:{limit}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # If we only have the name, search for the artist first
            if not artist_id and artist_name:
                results = self.sp.search(q=artist_name, type='artist', limit=1)
                
                if not results['artists']['items']:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Artist '{artist_name}' not found"
                    )
                    
                artist_id = results['artists']['items'][0]['id']
            
            # Get top tracks
            top_tracks = self.sp.artist_top_tracks(artist_id)
            result = [self._format_track(track) for track in top_tracks['tracks'][:limit]]
            
            # Cache the result
            self.cache.set(cache_key, result)
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Top tracks error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error getting top tracks: {str(e)}"
            )

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Spotify Track API",
    description="A fast and efficient API for retrieving Spotify track information",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
token_manager = TokenManager()

try:
    spotify_service = SpotifyService()
except Exception as e:
    logger.error(f"Failed to initialize Spotify service: {e}")
    spotify_service = None

# Security
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
    """Health check endpoint"""
    return {
        "status": "online",
        "api_version": "2.0.0",
        "spotify_service": "available" if spotify_service else "unavailable"
    }

@app.post("/token")
async def generate_token(request: Request):
    """Generate a new access token"""
    client_ip = request.client.host
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    return {
        "token": token_manager.generate_token(client_ip, user_agent),
        "expires_in": 86400  # 24 hours
    }

@app.post("/search", response_model=List[SpotifyTrack])
async def search_tracks(
    search_req: SearchRequest,
    token: str = Depends(validate_request)
):
    """
    Search for tracks based on query and type
    
    - query: Search text or URL (Spotify/YouTube)
    - search_type: track, album, artist, playlist, url
    - limit: Maximum number of results
    """
    if not spotify_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify service unavailable"
        )
    
    return await spotify_service.search_tracks(
        search_req.query,
        search_req.search_type,
        search_req.limit
    )

@app.post("/top-tracks", response_model=List[SpotifyTrack])
async def get_top_tracks(
    top_tracks_req: TopTracksRequest,
    token: str = Depends(validate_request)
):
    """
    Get top tracks for an artist
    
    - artist_id: Spotify artist ID
    - artist_name: Artist name (used if ID not provided)
    - limit: Maximum number of results
    """
    if not spotify_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify service unavailable"
        )
    
    return await spotify_service.get_artist_top_tracks(
        top_tracks_req.artist_id,
        top_tracks_req.artist_name,
        top_tracks_req.limit
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error"
        }
    )

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
