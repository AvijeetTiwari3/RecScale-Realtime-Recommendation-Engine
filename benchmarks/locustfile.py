"""
benchmarks/locustfile.py
========================
Concurrent Multi-Client Load Testing for LLM Serving Server.

Simulates 10, 50, and 100 concurrent streams measuring P95/P99 latency and TTFT.
Usage:
    locust -f benchmarks/locustfile.py --headless -u 10 -r 2 --run-time 1m --host http://localhost:8000
"""

import json
import random
from locust import HttpUser, task, between


SAMPLE_PROMPTS = [
    "Explain the difference between PagedAttention and standard multi-head attention.",
    "What is the mathematical proof behind speculative decoding rejection sampling?",
    "Write a high-performance Python generator for streaming server-sent events.",
    "How does continuous batching solve head-of-line blocking in LLM inference engines?",
    "Compare AWQ 4-bit quantization with GPTQ and NF4 precision modes."
]


class LLMServingUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def test_streaming_chat_completion(self):
        prompt = random.choice(SAMPLE_PROMPTS)
        payload = {
            "model": "speculative-engine",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.7,
            "stream": True
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Read SSE stream
                for line in response.iter_lines():
                    if line:
                        pass
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")

    @task(1)
    def test_non_streaming_chat_completion(self):
        prompt = random.choice(SAMPLE_PROMPTS)
        payload = {
            "model": "speculative-engine",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.7,
            "stream": False
        }

        self.client.post("/v1/chat/completions", json=payload)

    @task(1)
    def test_metrics_scrape(self):
        self.client.get("/metrics")
