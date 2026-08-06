import asyncio
import datetime
from typing import Any, Awaitable, Callable, Dict
import pandas as pd


class KPIStreamer:
    """
    WebSocket / Event subscription manager for real-time KPI metric updates.
    """

    def __init__(self):
        """Initializes the KPI streamer."""
        self._subscribers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}

    def subscribe(self, client_id: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Registers a WebSocket subscriber."""
        self._subscribers[client_id] = callback

    def unsubscribe(self, client_id: str) -> None:
        """Unregisters a WebSocket subscriber."""
        if client_id in self._subscribers:
            del self._subscribers[client_id]

    async def broadcast_kpi_update(self, metric_update: Dict[str, Any]) -> int:
        """Broadcasts metric update payload to all active subscribers.
        
        Args:
            metric_update: The metric update payload to broadcast.
            
        Returns:
            The number of active subscribers that received the broadcast.
        """
        count = 0
        tasks = []
        for client_id, callback in self._subscribers.items():
            tasks.append(callback(metric_update))
            count += 1
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        return count

    def get_active_subscribers_count(self) -> int:
        """Returns the number of active subscribers."""
        return len(self._subscribers)


class LiveKPIPublisher:
    """
    Publishes live KPI updates based on incoming streaming data.
    """

    def __init__(self, streamer: KPIStreamer):
        """Initializes the LiveKPIPublisher.
        
        Args:
            streamer: The KPIStreamer instance to use for broadcasting.
        """
        self.streamer = streamer

    async def process_new_data_event(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes latest KPI metrics from incoming streaming batch and broadcasts.
        
        Args:
            df: The incoming streaming batch dataframe.
            
        Returns:
            The constructed JSON update payload.
        """
        total_rows = len(df)
        revenue = float(df['revenue'].sum()) if 'revenue' in df.columns else 0.0
        
        payload = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'metrics': {
                'events_processed': total_rows,
                'total_revenue': revenue
            },
            'status': 'success'
        }
        
        await self.streamer.broadcast_kpi_update(payload)
        
        return payload
