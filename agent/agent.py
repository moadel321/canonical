from dotenv import load_dotenv
import os
import logging

from livekit.agents import JobContext, WorkerOptions, cli, metrics, MetricsCollectedEvent
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    silero,
)

from cached_llm import CachedLLM

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: JobContext):
    # Configure caching
    cache_enabled = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
    cache_endpoint = os.getenv("CACHE_ENDPOINT", "http://localhost:8000")
    
    # Initialize metrics collection
    usage_collector = metrics.UsageCollector()
    cache_latency_tracker = {
        "cached_responses": [],
        "non_cached_responses": [],
        "total_cache_hits": 0,
        "total_requests": 0
    }
    
    # Create base LLM
    base_llm = openai.LLM(model="gpt-4o-mini")
    
    # Wrap with caching if enabled
    if cache_enabled:
        logger.info(f"Semantic caching enabled, endpoint: {cache_endpoint}")
        llm_with_cache = CachedLLM(
            fallback_llm=base_llm,
            cache_endpoint=cache_endpoint,
            enable_caching=True,
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85")),
            cache_timeout=float(os.getenv("CACHE_TIMEOUT", "5.0"))
        )
        
        # Health check for cache
        try:
            health = await llm_with_cache.health_check()
            if health["cache_healthy"]:
                logger.info("Cache service is healthy and ready")
            else:
                logger.warning("Cache service is not healthy, will use fallback only")
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
        
        llm_instance = llm_with_cache
    else:
        logger.info("Semantic caching disabled, using direct LLM")
        llm_instance = base_llm
    
    session = AgentSession(
        stt=cartesia.STT(
            model="ink-whisper"
        ),
        llm=llm_instance,
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
    
    # Metrics event listener
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        # Collect standard metrics
        usage_collector.collect(ev.metrics)
        
        # Log all metrics for debugging
        metrics.log_metrics(ev.metrics)
        
        # Track LLM latency specifically for cache comparison
        if isinstance(ev.metrics, metrics.LLMMetrics):
            cache_latency_tracker["total_requests"] += 1
            
            # Check if this was a cached response
            if hasattr(llm_instance, '_cache_stats'):
                recent_hits = llm_instance._cache_stats.get("hits", 0)
                if recent_hits > cache_latency_tracker["total_cache_hits"]:
                    # This was a cache hit
                    cache_latency_tracker["total_cache_hits"] = recent_hits
                    cache_latency_tracker["cached_responses"].append({
                        "duration": ev.metrics.duration,
                        "ttft": ev.metrics.ttft,
                        "tokens_per_second": ev.metrics.tokens_per_second,
                        "timestamp": ev.metrics.timestamp if hasattr(ev.metrics, 'timestamp') else None
                    })
                    logger.info(f"CACHED LLM Response - Duration: {ev.metrics.duration:.3f}s, "
                              f"TTFT: {ev.metrics.ttft:.3f}s, TPS: {ev.metrics.tokens_per_second:.1f}")
                else:
                    # This was a cache miss
                    cache_latency_tracker["non_cached_responses"].append({
                        "duration": ev.metrics.duration,
                        "ttft": ev.metrics.ttft,
                        "tokens_per_second": ev.metrics.tokens_per_second,
                        "timestamp": ev.metrics.timestamp if hasattr(ev.metrics, 'timestamp') else None
                    })
                    logger.info(f"NON-CACHED LLM Response - Duration: {ev.metrics.duration:.3f}s, "
                              f"TTFT: {ev.metrics.ttft:.3f}s, TPS: {ev.metrics.tokens_per_second:.1f}")
            else:
                # No caching, all responses are non-cached
                cache_latency_tracker["non_cached_responses"].append({
                    "duration": ev.metrics.duration,
                    "ttft": ev.metrics.ttft,
                    "tokens_per_second": ev.metrics.tokens_per_second,
                    "timestamp": ev.metrics.timestamp if hasattr(ev.metrics, 'timestamp') else None
                })
                logger.info(f"DIRECT LLM Response - Duration: {ev.metrics.duration:.3f}s, "
                          f"TTFT: {ev.metrics.ttft:.3f}s, TPS: {ev.metrics.tokens_per_second:.1f}")
    
    # Latency comparison logging function
    async def log_latency_comparison():
        try:
            cached_count = len(cache_latency_tracker["cached_responses"])
            non_cached_count = len(cache_latency_tracker["non_cached_responses"])
            
            if cached_count > 0 and non_cached_count > 0:
                # Calculate averages
                avg_cached_duration = sum(r["duration"] for r in cache_latency_tracker["cached_responses"]) / cached_count
                avg_non_cached_duration = sum(r["duration"] for r in cache_latency_tracker["non_cached_responses"]) / non_cached_count
                
                avg_cached_ttft = sum(r["ttft"] for r in cache_latency_tracker["cached_responses"]) / cached_count
                avg_non_cached_ttft = sum(r["ttft"] for r in cache_latency_tracker["non_cached_responses"]) / non_cached_count
                
                latency_improvement = ((avg_non_cached_duration - avg_cached_duration) / avg_non_cached_duration) * 100
                ttft_improvement = ((avg_non_cached_ttft - avg_cached_ttft) / avg_non_cached_ttft) * 100
                
                logger.info("=== CACHE LATENCY COMPARISON ===")
                logger.info(f"Cached responses: {cached_count}, Avg duration: {avg_cached_duration:.3f}s, Avg TTFT: {avg_cached_ttft:.3f}s")
                logger.info(f"Non-cached responses: {non_cached_count}, Avg duration: {avg_non_cached_duration:.3f}s, Avg TTFT: {avg_non_cached_ttft:.3f}s")
                logger.info(f"LATENCY IMPROVEMENT: {latency_improvement:.1f}% duration, {ttft_improvement:.1f}% TTFT")
                
            # Log usage summary
            summary = usage_collector.get_summary()
            logger.info(f"Session Usage Summary: {summary}")
            
            # Log cache statistics if available
            if cache_enabled and isinstance(llm_instance, CachedLLM):
                cache_stats = await llm_instance.get_cache_stats()
                logger.info(f"Final Cache Statistics: {cache_stats}")
                
        except Exception as e:
            logger.error(f"Failed to log latency comparison: {e}")
    
    # Register shutdown callback for final metrics
    ctx.add_shutdown_callback(log_latency_comparison)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )
    
    # Log cache statistics if caching was enabled
    if cache_enabled and isinstance(llm_instance, CachedLLM):
        try:
            cache_stats = await llm_instance.get_cache_stats()
            logger.info(f"Session completed. Cache statistics: {cache_stats}")
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
        
        # Clean up cached LLM resources
        try:
            await llm_instance.aclose()
        except Exception as e:
            logger.error(f"Failed to close cached LLM: {e}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))