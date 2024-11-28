import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import time
import json
from pathlib import Path
import uvicorn
from rich.console import Console
from models.builder import ModelBuilder
from utils.memory import MemoryOptimizer
from utils.metrics import MetricsTracker

console = Console()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class ModelServer:
    def __init__(self, model_path: Path, port: int = 8000):
        self.app = FastAPI(title="MLX Model Server")
        self.port = port
        self.model_path = model_path
        self.model = None
        self.metrics = MetricsTracker()
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/v1/generate")
        async def generate(request: InferenceRequest):
            if not self.model:
                raise HTTPException(500, "Model not loaded")
                
            if request.stream:
                return StreamingResponse(
                    self._generate_stream(
                        request.prompt,
                        request.max_tokens,
                        request.temperature,
                        request.top_p
                    ),
                    media_type="text/event-stream"
                )
            
            response = self._generate_text(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p
            )
            return JSONResponse(response)
            
        @self.app.get("/v1/metrics")
        async def metrics():
            """Get server performance metrics"""
            return {
                "throughput": self.metrics.get_metric("tokens_per_second"),
                "latency_ms": self.metrics.get_metric("latency_ms"),
                "memory_usage": self._get_memory_usage(),
                "total_requests": self.metrics.get_metric("total_requests"),
                "active_requests": self.metrics.get_metric("active_requests")
            }
        
        @self.app.get("/v1/health")
        async def health():
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "device": "metal" if mx.metal.is_available() else "cpu",
                "memory_usage": self._get_memory_usage()
            }

    def _get_memory_usage(self):
        """Get current memory usage"""
        if mx.metal.is_available():
            return {
                "peak_gb": mx.metal.get_peak_memory() / 1e9,
                "current_gb": mx.metal.get_current_memory() / 1e9
            }
        return {}

    async def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        """Stream generation with performance tracking"""
        try:
            self.metrics.increment("active_requests")
            self.metrics.increment("total_requests")
            
            start_time = time.time()
            tokens_generated = 0
            
            # Initialize generation
            for token in self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            ):
                # Track performance
                tokens_generated += 1
                elapsed = time.time() - start_time
                self.metrics.update({
                    "tokens_per_second": tokens_generated / elapsed,
                    "latency_ms": (elapsed * 1000) / tokens_generated
                })
                
                # Stream token
                text = self.tokenizer.decode([token])
                yield f"data: {json.dumps({'text': text, 'done': False})}\n\n"
                
                # Force evaluation for accurate timing
                mx.eval(token)
            
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.metrics.decrement("active_requests")
    
    def serve(self):
        """Start the model server"""
        console.print(f"""
[bold blue]🚀 MLX Model Server[/bold blue]
Running on: http://localhost:{self.port}

API endpoints:
- POST /v1/generate  - Generate text
- GET  /v1/metrics   - Performance metrics
- GET  /v1/health    - Server health

Performance monitoring enabled:
- Throughput (tokens/sec)
- Latency (ms/token)
- Memory usage
- Request counts
        """)
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port) 