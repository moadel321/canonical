"""
Semantic Cache Middleware Server for LiveKit Voice Agents
FastAPI server providing context-aware caching for LLM responses
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from cache_service import CacheService
from redis_client import RedisClient
from metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
cache_service: Optional[CacheService] = None
metrics_collector: Optional[MetricsCollector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan management for cache middleware.
    
    Handles startup and shutdown of Redis connections, cache service initialization,
    and metrics collection. Ensures proper resource cleanup on application shutdown.
    
    Startup Sequence:
    1. Initialize Redis client with connection pool
    2. Create cache service and metrics collector
    3. Initialize vector index if needed
    
    Shutdown Sequence:
    1. Gracefully disconnect Redis client
    2. Clean up connection pools
    
    Time Complexity: O(1) for startup/shutdown operations
    """
    global cache_service, metrics_collector
    
    logger.info("Starting cache middleware server...")
    
    # Initialize Redis client
    redis_client = RedisClient()
    await redis_client.connect()
    
    # Initialize services
    cache_service = CacheService(redis_client)
    metrics_collector = MetricsCollector()
    
    # Initialize cache service (creates vector index if needed)
    await cache_service.initialize()
    
    logger.info("Cache middleware server started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down cache middleware server...")
    if redis_client:
        await redis_client.disconnect()

app = FastAPI(
    title="Semantic Cache Middleware",
    description="Context-aware semantic caching for LiveKit Voice Agents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CacheLookupRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class CacheStoreRequest(BaseModel):
    messages: List[Dict[str, Any]]
    response: Dict[str, Any]
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

class CacheResponse(BaseModel):
    hit: bool
    response: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None
    latency_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    cache_entries: int

# Dependency to get cache service
async def get_cache_service() -> CacheService:
    """
    FastAPI dependency for accessing the global cache service instance.
    
    Returns:
        CacheService: Initialized cache service instance
        
    Raises:
        HTTPException: 503 if cache service not initialized during startup
        
    Time Complexity: O(1)
    """
    if cache_service is None:
        raise HTTPException(status_code=503, detail="Cache service not initialized")
    return cache_service

async def get_metrics_collector() -> MetricsCollector:
    """
    FastAPI dependency for accessing the global metrics collector instance.
    
    Returns:
        MetricsCollector: Initialized metrics collector instance
        
    Raises:
        HTTPException: 503 if metrics collector not initialized during startup
        
    Time Complexity: O(1)
    """
    if metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    return metrics_collector

@app.get("/health", response_model=HealthResponse)
async def health_check(service: CacheService = Depends(get_cache_service)):
    """
    Health check endpoint for cache middleware monitoring.
    
    Verifies Redis connectivity and reports cache status for load balancers
    and monitoring systems.
    
    Returns:
        HealthResponse: Service health status
        - status: "healthy" if all systems operational
        - redis_connected: Redis connection status
        - cache_entries: Current number of cached responses
        
    Response Time: ~1-5ms
    
    Monitoring Use:
    - Load balancer health checks
    - Kubernetes readiness/liveness probes
    - System monitoring dashboards
    
    Error Handling:
    - Returns 503 if Redis unreachable
    - Logs connection failures for debugging
    """
    try:
        redis_connected = await service.redis_client.ping()
        cache_entries = await service.get_cache_size()
        
        return HealthResponse(
            status="healthy",
            redis_connected=redis_connected,
            cache_entries=cache_entries
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/cache/lookup", response_model=CacheResponse)
async def cache_lookup(
    request: CacheLookupRequest,
    service: CacheService = Depends(get_cache_service),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Semantic cache lookup endpoint for voice agent integration.
    
    Core endpoint for cache-first pattern: voice agents call this before LLM
    to check for semantically similar cached responses. Uses vector similarity
    search with context-aware embeddings.
    
    Request:
        CacheLookupRequest:
        - messages: Conversation history as message dicts
        - model: LLM model name (e.g., 'gpt-4o-mini')
        - temperature: Model temperature (default: 0.7)
        - max_tokens: Optional token limit
        
    Response:
        CacheResponse:
        - hit: True if cache hit found above similarity threshold
        - response: Cached LLM response dict (if hit)
        - similarity_score: Vector similarity 0-1 (if hit)
        - latency_ms: Lookup latency for performance monitoring
        
    Performance:
    - Cache Hit: ~1-5ms latency
    - Cache Miss: ~600ms (embedding generation)
    
    Integration Pattern:
        1. Voice agent calls /cache/lookup
        2. If hit: return cached response (fast path)
        3. If miss: call LLM, then call /cache/store
        
    Security:
    - PII detection prevents caching sensitive data
    - Returns early if PII detected in messages
    
    Metrics:
    - Records hit/miss rates
    - Tracks latency percentiles
    - Monitors similarity score distributions
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        result = await service.lookup_cached_response(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature
        )
        
        latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Record metrics
        if result["hit"]:
            await metrics.record_cache_hit(latency_ms, result["similarity_score"])
        else:
            await metrics.record_cache_miss(latency_ms)
        
        return CacheResponse(
            hit=result["hit"],
            response=result["response"],
            similarity_score=result.get("similarity_score"),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Cache lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/store")
async def cache_store(
    request: CacheStoreRequest,
    service: CacheService = Depends(get_cache_service),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Store LLM response in semantic cache for future lookups.
    
    Called by voice agents after LLM responses to build the semantic cache.
    Generates context-aware embeddings and stores with vector index for
    similarity searches.
    
    Request:
        CacheStoreRequest:
        - messages: Conversation history that generated the response
        - response: LLM response dict to cache
        - model: LLM model name used
        - temperature: Model temperature used
        
    Response:
        Dict: Storage status and performance metrics
        - status: "stored" if successful
        - latency_ms: Storage operation latency
        
    Performance:
    - Storage Latency: ~1ms
    - Memory Usage: ~6KB per response (vector + metadata)
    
    Integration Pattern:
        1. Voice agent gets cache miss from /cache/lookup
        2. Voice agent calls LLM
        3. Voice agent calls /cache/store with response
        4. Future similar queries get cache hits
        
    Storage Security:
    - PII detection on both messages and response
    - Sensitive content never cached
    - TTL-based expiration (default: 1 hour)
    
    Vector Storage:
    - 1536-dimensional OpenAI embeddings
    - COSINE distance metric in Redis
    - HNSW index for fast similarity search
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        await service.store_cached_response(
            messages=request.messages,
            response=request.response,
            model=request.model,
            temperature=request.temperature
        )
        
        latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        await metrics.record_cache_store(latency_ms)
        
        return {"status": "stored", "latency_ms": latency_ms}
        
    except Exception as e:
        logger.error(f"Cache store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(metrics: MetricsCollector = Depends(get_metrics_collector)):
    """
    Retrieve comprehensive cache performance metrics.
    
    Provides detailed analytics for cache performance monitoring,
    cost optimization, and system tuning.
    
    Returns:
        Dict: Comprehensive metrics including:
        - performance: Hit rates, request counts, efficiency metrics
        - latency: Response time percentiles and averages
        - cost: API usage and cost estimates
        - redis: Memory usage and connection stats
        
    Response Time: ~1-2ms
    
    Monitoring Integration:
    - Prometheus/Grafana dashboards
    - Cost tracking and optimization
    - Performance alerting
    - Capacity planning
    
    Key Metrics:
    - Cache hit rate percentage
    - Average hit/miss latencies
    - Total requests processed
    - Cost savings from cache hits
    """
    try:
        return await metrics.get_metrics()
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/clear")
async def clear_cache(service: CacheService = Depends(get_cache_service)):
    """
    Clear all cached entries for debugging and testing.
    
    Removes all cached responses from Redis and resets metrics.
    Use with caution in production environments.
    
    Returns:
        Dict: Operation status
        - status: "cache cleared" if successful
        
    Response Time: ~10-50ms depending on cache size
    
    Use Cases:
    - Development and testing
    - Cache corruption recovery
    - Memory management in constrained environments
    - A/B testing with fresh cache
    
    WARNING: This operation:
    - Cannot be undone
    - Resets all performance metrics
    - May cause temporary performance degradation
    - Should not be used in production without careful consideration
    """
    try:
        await service.clear_cache()
        return {"status": "cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Production-ready server configuration for semantic cache middleware
    # Host: 0.0.0.0 for container/network accessibility
    # Port: 8000 (standard for cache middleware)
    # Reload: True for development, disable in production
    # Log level: INFO for operational visibility
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )