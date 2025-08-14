from ..core.ports import EventBus

class InProcEventBus(EventBus):
    def publish(self, topic: str, event: object) -> None:
        # No-op for now; swap with Redis/Kafka later
        return
