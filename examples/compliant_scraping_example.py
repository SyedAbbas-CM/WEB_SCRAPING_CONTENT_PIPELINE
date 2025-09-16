#!/usr/bin/env python3
"""
Compliant Scraping Example
Demonstrates how to use the ScrapeHive system with proper API-first approach
"""

import os
import sys
import time
import json
from typing import List, Dict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.unified_platform_manager import (
    UnifiedPlatformManager, Platform, SearchRequest, AggregatedResult
)
from scrapers.reddit_api_client import RedditAPIClient
from scrapers.youtube_api_client import YouTubeAPIClient


def setup_environment():
    """Setup environment for compliant scraping"""
    # Set compliance mode to API-first for this example
    os.environ['COMPLIANCE_MODE'] = 'api_first'

    # You would set your API keys here:
    # os.environ['REDDIT_CLIENT_ID'] = 'your_reddit_client_id'
    # os.environ['REDDIT_CLIENT_SECRET'] = 'your_reddit_client_secret'
    # os.environ['YOUTUBE_API_KEY'] = 'your_youtube_api_key'

    print("🔧 Environment configured for compliant scraping")
    print(f"📋 Compliance mode: {os.environ.get('COMPLIANCE_MODE', 'api_first')}")


def example_single_platform_search():
    """Example: Search a single platform using API"""
    print("\n" + "="*60)
    print("🎯 Single Platform Search Example")
    print("="*60)

    manager = UnifiedPlatformManager()

    try:
        # Search Reddit for programming content
        result = manager.search_single_platform(
            platform=Platform.REDDIT,
            query="python programming",
            limit=5
        )

        if result.success:
            print(f"✅ Reddit search successful!")
            print(f"📊 Method used: {result.method_used}")
            print(f"📈 Found {result.data.get('count', 0)} results")

            # Show first few results
            posts = result.data.get('posts', [])
            for i, post in enumerate(posts[:3], 1):
                print(f"\n{i}. {post.get('title', 'No title')[:60]}...")
                print(f"   👤 Author: {post.get('author', 'Unknown')}")
                print(f"   📊 Score: {post.get('score', 0)}")
                print(f"   🏠 Subreddit: {post.get('subreddit', 'Unknown')}")
        else:
            print(f"❌ Reddit search failed: {result.error_message}")

    except Exception as e:
        print(f"💥 Error during search: {e}")


def example_multi_platform_search():
    """Example: Search multiple platforms simultaneously"""
    print("\n" + "="*60)
    print("🌐 Multi-Platform Search Example")
    print("="*60)

    manager = UnifiedPlatformManager()

    try:
        platforms = [Platform.REDDIT, Platform.YOUTUBE]

        result = manager.search_multiple_platforms(
            platforms=platforms,
            query="machine learning",
            limit_per_platform=3
        )

        print(f"🎯 Searched platforms: {', '.join(result.platforms_used)}")
        print(f"📊 Total items found: {result.total_items}")
        print(f"🔧 Methods used: {json.dumps(result.methods_used, indent=2)}")

        if result.compliance_notes:
            print(f"⚖️  Compliance notes:")
            for note in result.compliance_notes:
                print(f"   • {note}")

        # Show results from each platform
        for platform, data in result.results.items():
            print(f"\n📱 {platform.upper()} Results:")

            if platform == 'reddit' and 'posts' in data:
                for post in data['posts'][:2]:
                    print(f"  • {post.get('title', 'No title')[:50]}...")

            elif platform == 'youtube' and 'videos' in data:
                for video in data['videos'][:2]:
                    print(f"  • {video.get('title', 'No title')[:50]}...")
                    print(f"    📺 Channel: {video.get('channel_title', 'Unknown')}")

        # Show any errors
        if result.errors:
            print(f"\n❌ Errors encountered:")
            for platform, error in result.errors.items():
                print(f"   • {platform}: {error}")

    except Exception as e:
        print(f"💥 Error during multi-platform search: {e}")


def example_priority_based_search():
    """Example: Priority-based search with fallback strategy"""
    print("\n" + "="*60)
    print("🎯 Priority-Based Search with Fallback")
    print("="*60)

    manager = UnifiedPlatformManager()

    # Create search requests with different priorities
    search_requests = [
        SearchRequest(
            platform=Platform.REDDIT,
            query="artificial intelligence",
            limit=5,
            priority=1,  # High priority
            filters={'sort': 'top', 'time_filter': 'week'}
        ),
        SearchRequest(
            platform=Platform.YOUTUBE,
            query="AI tutorial",
            limit=3,
            priority=2,  # Medium priority
            filters={'order': 'relevance'}
        )
    ]

    try:
        result = manager.search_with_fallback_strategy(search_requests)

        print(f"🎯 Processed {len(search_requests)} search requests")
        print(f"✅ Successful platforms: {', '.join(result.platforms_used)}")
        print(f"📊 Total results: {result.total_items}")

        # Show compliance information
        if result.compliance_notes:
            print(f"\n⚖️  Compliance Information:")
            for note in result.compliance_notes:
                print(f"   • {note}")

        # Show method usage
        print(f"\n🔧 Methods Used:")
        for platform, method in result.methods_used.items():
            print(f"   • {platform}: {method}")

    except Exception as e:
        print(f"💥 Error during priority search: {e}")


def example_rate_limit_handling():
    """Example: Demonstrate rate limit handling"""
    print("\n" + "="*60)
    print("⏱️  Rate Limit Handling Example")
    print("="*60)

    manager = UnifiedPlatformManager()

    # Make multiple rapid requests to show rate limiting
    for i in range(3):
        print(f"\n🔄 Request {i+1}/3...")

        try:
            result = manager.search_single_platform(
                platform=Platform.REDDIT,
                query=f"test query {i+1}",
                limit=2
            )

            if result.success:
                print(f"✅ Success - Method: {result.method_used}")
                print(f"📊 Found {result.data.get('count', 0)} results")
            else:
                print(f"❌ Failed: {result.error_message}")

        except Exception as e:
            print(f"💥 Error: {e}")

        # Small delay between requests
        time.sleep(1)


def example_statistics_monitoring():
    """Example: Monitor statistics and compliance metrics"""
    print("\n" + "="*60)
    print("📊 Statistics and Monitoring Example")
    print("="*60)

    manager = UnifiedPlatformManager()

    # Perform a few operations
    manager.search_single_platform(Platform.REDDIT, "test", limit=2)

    try:
        # Get global statistics
        stats = manager.get_global_stats()

        print(f"📈 Global Statistics:")
        print(f"   • Total requests: {stats.get('total_requests', 0)}")
        print(f"   • Successful: {stats.get('successful_requests', 0)}")
        print(f"   • Failed: {stats.get('failed_requests', 0)}")
        print(f"   • API requests: {stats.get('api_requests', 0)}")
        print(f"   • Fallback requests: {stats.get('fallback_requests', 0)}")
        print(f"   • Compliance mode: {stats.get('compliance_mode', 'unknown')}")

        # Show platform-specific stats
        platform_stats = stats.get('platform_stats', {})
        for platform, p_stats in platform_stats.items():
            print(f"\n📱 {platform.upper()} Statistics:")
            print(f"   • API calls: {p_stats.get('api_calls', 0)}")
            print(f"   • Success rate: {p_stats.get('api_success', 0)}/{p_stats.get('api_calls', 0)}")

        # Show rate limiter status
        rate_limiter_status = stats.get('global_rate_limiter', {})
        if rate_limiter_status:
            print(f"\n⏱️  Rate Limiter Status:")
            for platform, r_stats in rate_limiter_status.items():
                print(f"   • {platform}: {r_stats.get('requests_last_hour', 0)}/{r_stats.get('limit', 0)} requests")

    except Exception as e:
        print(f"💥 Error getting statistics: {e}")


def example_compliance_modes():
    """Example: Demonstrate different compliance modes"""
    print("\n" + "="*60)
    print("⚖️  Compliance Modes Example")
    print("="*60)

    compliance_modes = ['api_only', 'api_first', 'balanced']

    for mode in compliance_modes:
        print(f"\n🔧 Testing compliance mode: {mode}")
        os.environ['COMPLIANCE_MODE'] = mode

        try:
            manager = UnifiedPlatformManager()

            result = manager.search_single_platform(
                platform=Platform.REDDIT,
                query="test",
                limit=2
            )

            if result.success:
                print(f"✅ Mode '{mode}' - Success with method: {result.method_used}")
            else:
                print(f"❌ Mode '{mode}' - Failed: {result.error_message}")

        except Exception as e:
            print(f"💥 Mode '{mode}' - Error: {e}")

    # Reset to default
    os.environ['COMPLIANCE_MODE'] = 'api_first'


def main():
    """Main demonstration function"""
    print("🕷️ ScrapeHive Compliant Scraping Examples")
    print("=" * 60)

    setup_environment()

    # Check if API credentials are available
    has_reddit_creds = bool(os.getenv('REDDIT_CLIENT_ID') and os.getenv('REDDIT_CLIENT_SECRET'))
    has_youtube_creds = bool(os.getenv('YOUTUBE_API_KEY'))

    if not (has_reddit_creds or has_youtube_creds):
        print("\n⚠️  WARNING: No API credentials found!")
        print("   Set your API keys in environment variables to see full functionality.")
        print("   The examples will demonstrate error handling and fallback behavior.")

    # Run examples
    try:
        example_single_platform_search()
        example_multi_platform_search()
        example_priority_based_search()
        example_rate_limit_handling()
        example_statistics_monitoring()
        example_compliance_modes()

    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")

    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("\n📚 Key Takeaways:")
    print("   • Always start with API-first approach")
    print("   • Respect rate limits and implement proper delays")
    print("   • Monitor compliance and adjust behavior accordingly")
    print("   • Use fallback methods only when necessary")
    print("   • Track statistics for optimization")
    print("\n⚖️  Remember: Ethical scraping respects platform terms of service!")


if __name__ == "__main__":
    main()