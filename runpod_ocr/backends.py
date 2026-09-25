"""GPU inference backends: one scale-to-zero vLLM endpoint per VLM.

Flash provisions these in image mode (`runpod-ocr backend up`) and tracks them
by name in `.flash/resources.pkl`. The OCR worker reaches them through Runpod's
OpenAI-compatible passthrough.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from runpod_flash import CudaVersion, Endpoint, GpuGroup

from .schema import ModelName

# vLLM 0.28 on CUDA 13.0
VLLM_WORKER_IMAGE = "runpod/worker-v1-vllm:v2.27.1"


@dataclass(frozen=True, kw_only=True)
class VllmBackend:
    name: str
    hf_repo: str
    # env var holding the provisioned endpoint id, read by the OCR worker
    endpoint_id_env: str
    # extra worker-vllm env vars; each becomes a `vllm serve` flag
    vllm_env: Mapping[str, str] = field(default_factory=dict)

    def endpoint(self) -> Endpoint:
        return Endpoint(
            name=self.name,
            image=VLLM_WORKER_IMAGE,
            # ~1B models fit easily in 24 GB; A5000/3090/L4 first, 4090 as fallback
            gpu=[GpuGroup.AMPERE_24, GpuGroup.ADA_24],
            workers=(0, 3),
            idle_timeout=60,
            execution_timeout_ms=600_000,
            min_cuda_version=CudaVersion.V13_0,
            env={"MODEL_NAME": self.hf_repo, "MAX_MODEL_LEN": "16384", **self.vllm_env},
        )

    @staticmethod
    def openai_url(endpoint_id: str) -> str:
        return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"


BACKENDS: dict[ModelName, VllmBackend] = {
    "paddleocr-vl-1.6": VllmBackend(
        name="ocr-paddleocr-vl-1-6",
        hf_repo="PaddlePaddle/PaddleOCR-VL-1.6",
        endpoint_id_env="PADDLEOCR_VL_ENDPOINT_ID",
        # flags from the vLLM PaddleOCR-VL recipe
        vllm_env={
            "TRUST_REMOTE_CODE": "true",
            "MAX_NUM_BATCHED_TOKENS": "16384",
            "ENABLE_PREFIX_CACHING": "false",
            "MM_PROCESSOR_CACHE_GB": "0",
        },
    ),
}
