import pytest
import asyncio
from analytics.realtime import KPIStreamer, LiveKPIPublisher

@pytest.mark.asyncio
async def test_kpi_streamer_subscription():
    streamer = KPIStreamer()
    
    received_data = []
    
    async def mock_callback(data):
        received_data.append(data)
        
    streamer.subscribe("client1", mock_callback)
    assert streamer.get_active_subscribers_count() == 1
    
    await streamer.broadcast_kpi_update({"metric": "value"})
    assert len(received_data) == 1
    assert received_data[0]["metric"] == "value"
    
    streamer.unsubscribe("client1")
    assert streamer.get_active_subscribers_count() == 0

@pytest.mark.asyncio
async def test_live_kpi_publisher(sample_kpi_data):
    streamer = KPIStreamer()
    publisher = LiveKPIPublisher(streamer)
    
    received_data = []
    
    async def mock_callback(data):
        received_data.append(data)
        
    streamer.subscribe("client1", mock_callback)
    
    payload = await publisher.process_new_data_event(sample_kpi_data)
    assert payload['metrics']['events_processed'] == 5
    assert payload['metrics']['total_revenue'] == 800.0
    
    assert len(received_data) == 1
    assert received_data[0] == payload
