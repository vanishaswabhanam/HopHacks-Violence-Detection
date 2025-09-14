"""
Discord Webhook Notification System
Real-time alerts for violence detection system
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import aiohttp

class DiscordNotifier:
    """Discord webhook notification system for security alerts"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE"
        self.enabled = webhook_url is not None and "YOUR_WEBHOOK_URL_HERE" not in webhook_url
        
        # Alert thresholds
        self.threat_levels = {
            'low': {'color': 0x00ff00, 'emoji': '🟢', 'priority': 1},
            'medium': {'color': 0xffaa00, 'emoji': '🟡', 'priority': 2}, 
            'high': {'color': 0xff4444, 'emoji': '🔴', 'priority': 3},
            'critical': {'color': 0x8b0000, 'emoji': '🚨', 'priority': 4}
        }
        
        # Camera locations for context
        self.camera_locations = {
            'cam_1': 'Hallway A (North Wing)',
            'cam_2': 'Cafeteria (Main Floor)', 
            'cam_3': 'Main Entrance (Ground Floor)',
            'cam_4': 'Gymnasium (East Wing)'
        }
    
    def get_threat_color(self, threat_type: str, confidence: float) -> int:
        """Get Discord embed color based on threat type and confidence"""
        if threat_type in ['fighting', 'violence', 'assault', 'shooting']:
            return self.threat_levels['critical']['color']
        elif threat_type in ['robbery', 'burglary', 'arrest']:
            return self.threat_levels['high']['color']
        elif threat_type in ['suspicious', 'abuse', 'stealing']:
            return self.threat_levels['medium']['color']
        else:
            return self.threat_levels['low']['color']
    
    def get_threat_emoji(self, threat_type: str) -> str:
        """Get emoji for threat type"""
        threat_emojis = {
            'fighting': '👊',
            'violence': '⚔️',
            'robbery': '💰',
            'burglary': '🏠',
            'arrest': '🚔',
            'suspicious': '👀',
            'abuse': '😡',
            'stealing': '🤏',
            'normal': '✅',
            'shooting': '🔫',
            'assault': '💥'
        }
        return threat_emojis.get(threat_type, '⚠️')
    
    def create_alert_embed(self, detection: Dict) -> Dict:
        """Create Discord embed for detection alert"""
        threat_type = detection.get('threat_type', 'unknown')
        confidence = detection.get('confidence', 0.0)
        camera_id = detection.get('camera_id', 'unknown')
        timestamp = detection.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        indicators = detection.get('indicators', [])
        
        # Get camera location
        camera_location = self.camera_locations.get(camera_id, f"Camera {camera_id}")
        
        # Create embed
        embed = {
            "title": f"{self.get_threat_emoji(threat_type)} Security Alert",
            "description": f"**{threat_type.title()}** detected at **{camera_location}**",
            "color": self.get_threat_color(threat_type, confidence),
            "timestamp": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "🎯 Threat Type",
                    "value": threat_type.title(),
                    "inline": True
                },
                {
                    "name": "📊 Confidence",
                    "value": f"{confidence:.1%}",
                    "inline": True
                },
                {
                    "name": "📍 Location",
                    "value": camera_location,
                    "inline": True
                },
                {
                    "name": "🕐 Time",
                    "value": timestamp,
                    "inline": True
                },
                {
                    "name": "📹 Camera ID",
                    "value": camera_id,
                    "inline": True
                },
                {
                    "name": "🔍 Status",
                    "value": "Active Monitoring",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Violence Detection System • Real-time Monitoring"
            }
        }
        
        # Add indicators if available
        if indicators:
            indicator_text = ", ".join(indicators[:3])  # Show first 3 indicators
            if len(indicators) > 3:
                indicator_text += f" (+{len(indicators) - 3} more)"
            embed["fields"].append({
                "name": "🚨 Indicators",
                "value": indicator_text,
                "inline": False
            })
        
        return embed
    
    def create_summary_embed(self, stats: Dict) -> Dict:
        """Create Discord embed for system status summary"""
        return {
            "title": "📊 Security System Status",
            "description": "Daily monitoring summary",
            "color": 0x0099ff,
            "timestamp": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "📹 Cameras Online",
                    "value": f"{stats.get('cameras_online', 0)}/{stats.get('total_cameras', 0)}",
                    "inline": True
                },
                {
                    "name": "🚨 Alerts Today",
                    "value": str(stats.get('alerts_today', 0)),
                    "inline": True
                },
                {
                    "name": "⏱️ System Uptime",
                    "value": f"{stats.get('system_uptime', 0)}%",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Violence Detection System • Automated Report"
            }
        }
    
    async def send_alert(self, detection: Dict) -> bool:
        """Send detection alert to Discord"""
        if not self.enabled:
            print(f"🔕 Discord notifications disabled - would send: {detection.get('threat_type')} at {detection.get('camera_id')}")
            return False
        
        try:
            embed = self.create_alert_embed(detection)
            
            payload = {
                "embeds": [embed],
                "username": "Security Bot",
                "avatar_url": "https://cdn.discordapp.com/emojis/1234567890123456789.png"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        print(f"✅ Discord alert sent: {detection.get('threat_type')} at {detection.get('camera_id')}")
                        return True
                    else:
                        print(f"❌ Discord alert failed: {response.status}")
                        return False
        
        except Exception as e:
            print(f"❌ Discord notification error: {e}")
            return False
    
    async def send_summary(self, stats: Dict) -> bool:
        """Send system status summary to Discord"""
        if not self.enabled:
            print("🔕 Discord notifications disabled - would send summary")
            return False
        
        try:
            embed = self.create_summary_embed(stats)
            
            payload = {
                "embeds": [embed],
                "username": "Security Bot",
                "avatar_url": "https://cdn.discordapp.com/emojis/1234567890123456789.png"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        print("✅ Discord summary sent")
                        return True
                    else:
                        print(f"❌ Discord summary failed: {response.status}")
                        return False
        
        except Exception as e:
            print(f"❌ Discord summary error: {e}")
            return False
    
    def send_alert_sync(self, detection: Dict) -> bool:
        """Synchronous version of send_alert for non-async contexts"""
        if not self.enabled:
            print(f"🔕 Discord notifications disabled - would send: {detection.get('threat_type')} at {detection.get('camera_id')}")
            return False
        
        try:
            embed = self.create_alert_embed(detection)
            
            payload = {
                "embeds": [embed],
                "username": "Security Bot",
                "avatar_url": "https://cdn.discordapp.com/emojis/1234567890123456789.png"
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                print(f"✅ Discord alert sent: {detection.get('threat_type')} at {detection.get('camera_id')}")
                return True
            else:
                print(f"❌ Discord alert failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Discord notification error: {e}")
            return False
    
    def test_webhook(self) -> bool:
        """Test Discord webhook connection"""
        if not self.enabled:
            print("🔕 Discord webhook not configured")
            return False
        
        try:
            test_payload = {
                "content": "🧪 **Test Message** - Violence Detection System is online!",
                "username": "Security Bot"
            }
            
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            
            if response.status_code == 204:
                print("✅ Discord webhook test successful!")
                return True
            else:
                print(f"❌ Discord webhook test failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Discord webhook test error: {e}")
            return False

# Global Discord notifier instance
discord_notifier = DiscordNotifier()

def setup_discord_webhook(webhook_url: str) -> bool:
    """Setup Discord webhook URL"""
    global discord_notifier
    discord_notifier = DiscordNotifier(webhook_url)
    return discord_notifier.test_webhook()

async def send_discord_alert(detection: Dict) -> bool:
    """Send Discord alert for detection"""
    return await discord_notifier.send_alert(detection)

def send_discord_alert_sync(detection: Dict) -> bool:
    """Send Discord alert synchronously"""
    return discord_notifier.send_alert_sync(detection)

async def send_discord_summary(stats: Dict) -> bool:
    """Send Discord summary"""
    return await discord_notifier.send_summary(stats)

def test_discord_connection() -> bool:
    """Test Discord connection"""
    return discord_notifier.test_webhook()

# Test function
async def test_discord_notifications():
    """Test Discord notification system with sample data"""
    print("🧪 Testing Discord Notification System...")
    
    # Test webhook connection
    if not test_discord_connection():
        print("❌ Discord webhook not configured or failed")
        return
    
    # Test alert notification
    sample_detection = {
        "camera_id": "cam_1",
        "threat_type": "fighting",
        "confidence": 0.95,
        "timestamp": "16:45:30",
        "indicators": ["High motion detected", "Rapid color changes", "Multiple people"],
        "location": {"x": 150, "y": 200}
    }
    
    print("\n📤 Sending test alert...")
    await send_discord_alert(sample_detection)
    
    # Test summary notification
    sample_stats = {
        "cameras_online": 4,
        "total_cameras": 4,
        "alerts_today": 15,
        "system_uptime": 99.8
    }
    
    print("\n📊 Sending test summary...")
    await send_discord_summary(sample_stats)
    
    print("\n✅ Discord notification test completed!")

if __name__ == "__main__":
    asyncio.run(test_discord_notifications())
