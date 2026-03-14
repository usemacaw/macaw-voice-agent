"""
Shared gRPC server infrastructure for STT/TTS microservices.

Provides common server configuration (keepalive, health check, reflection,
graceful shutdown) to eliminate boilerplate duplication between servers.
"""

import asyncio
import logging
import os
import signal
from typing import Callable, Optional

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

logger = logging.getLogger(__name__)


def _default_grpc_options() -> list[tuple]:
    max_msg = int(os.getenv("GRPC_MAX_MESSAGE_SIZE", str(10 * 1024 * 1024)))
    return [
        ("grpc.max_send_message_length", max_msg),
        ("grpc.max_receive_message_length", max_msg),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.http2.max_ping_strikes", 0),
    ]


class GrpcMicroservice:
    """Base for gRPC microservices with health check, reflection, and graceful shutdown.

    Usage:
        micro = GrpcMicroservice("my.Service", port=50070)
        await micro.start(
            add_servicers=lambda server, health_svc: ...,
            service_names=("my.Service",),
            provider=my_provider,
        )
        await micro.wait()
    """

    def __init__(self, service_name: str, port: int | None = None):
        self._service_name = service_name
        self._port = port or int(os.getenv("GRPC_PORT", "50070"))
        self._host = os.getenv("GRPC_HOST", "0.0.0.0")
        self._server: Optional[grpc.aio.Server] = None
        self._health_servicer: Optional[health.HealthServicer] = None
        self._provider = None

    @property
    def health_servicer(self) -> Optional[health.HealthServicer]:
        return self._health_servicer

    async def start(
        self,
        add_servicers: Callable,
        service_names: tuple[str, ...],
        provider=None,
    ) -> None:
        """Start the gRPC server.

        Args:
            add_servicers: Callback(server, health_servicer) to register service-specific servicers.
            service_names: Tuple of fully qualified service names for reflection.
            provider: Optional provider instance for lifecycle management.
        """
        self._provider = provider
        self._server = grpc.aio.server(options=_default_grpc_options())
        self._health_servicer = health.HealthServicer()

        add_servicers(self._server, self._health_servicer)

        health_pb2_grpc.add_HealthServicer_to_server(self._health_servicer, self._server)

        all_service_names = (
            *service_names,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(all_service_names, self._server)

        self._health_servicer.set(
            self._service_name,
            health_pb2.HealthCheckResponse.SERVING,
        )

        self._server.add_insecure_port(f"{self._host}:{self._port}")
        await self._server.start()
        logger.info(f"gRPC server started on {self._host}:{self._port} ({self._service_name})")

    async def stop(self, grace: float = 5.0) -> None:
        if self._health_servicer:
            self._health_servicer.set(
                self._service_name,
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )
        if self._server:
            await self._server.stop(grace=grace)
        if self._provider and hasattr(self._provider, "disconnect"):
            await self._provider.disconnect()
        logger.info(f"gRPC server stopped ({self._service_name})")

    async def wait(self) -> None:
        if self._server:
            await self._server.wait_for_termination()

    async def run_until_signal(self) -> None:
        """Run until SIGINT/SIGTERM."""
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()
        logger.info("Shutting down...")
        await self.stop()
