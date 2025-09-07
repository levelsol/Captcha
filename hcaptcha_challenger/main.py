import asyncio
import json
from playwright.async_api import async_playwright
from hcaptcha_challenger import AgentV, AgentConfig, CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(page):
    """Automates the process of solving an hCaptcha challenge using Ollama/LLaVA."""
    print("🔧 Setting up hCaptcha challenger with Ollama...")
    
    # Initialize the Agent with Ollama configuration
    agent_config = AgentConfig(
        # Ollama Configuration
        OLLAMA_URL="http://localhost:11434",
        
        # Model Configuration - using LLaVA models
        IMAGE_CLASSIFIER_MODEL="llava:latest",
        CHALLENGE_CLASSIFIER_MODEL="llava:latest",
        SPATIAL_POINT_REASONER_MODEL="llava:latest",
        SPATIAL_PATH_REASONER_MODEL="llava:latest",
        
        # Enable debug mode to see what's happening
        enable_challenger_debug=True,
        
        # Timeout settings (increase if your local model is slow)
        EXECUTION_TIMEOUT=180,  # 3 minutes for local processing
        RESPONSE_TIMEOUT=60,    # 1 minute response timeout
        
        # Keep retries enabled
        RETRY_ON_FAILURE=True,
    )
    
    print("✅ Ollama configuration loaded")
    print(f"📡 Using Ollama at: {agent_config.OLLAMA_URL}")
    print(f"🤖 Using model: {agent_config.IMAGE_CLASSIFIER_MODEL}")
    
    agent = AgentV(page=page, agent_config=agent_config)

    print("📱 Clicking checkbox to trigger challenge...")
    # Click the checkbox to trigger the challenge
    await agent.robotic_arm.click_checkbox()

    print("⏳ Waiting for challenge to appear and solving with local AI...")
    # Wait for the challenge to appear and be ready for solving
    result = await agent.wait_for_challenge()
    
    print(f"🎯 Challenge result: {result}")
    return agent


async def main():
    print("🚀 Starting hCaptcha Challenger Demo with Ollama/LLaVA")
    print("=" * 60)
    
    # Check if Ollama is accessible (basic check)
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    models = await response.json()
                    print("✅ Ollama is running and accessible")
                    
                    # Check if LLaVA model is available
                    model_names = [model.get('name', '') for model in models.get('models', [])]
                    llava_models = [name for name in model_names if 'llava' in name.lower()]
                    
                    if llava_models:
                        print(f"🎯 Available LLaVA models: {', '.join(llava_models)}")
                    else:
                        print("⚠️  Warning: No LLaVA models found. Please run: ollama pull llava:latest")
                else:
                    print("❌ Ollama is not responding correctly")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return
    
    async with async_playwright() as p:
        # Launch browser (you can set headless=True to hide browser)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        # Option 1: Use your specific demo URL
        demo_url = "https://accounts.hcaptcha.com/demo?sitekey=019f1553-3845-481c-a6f5-5a60ccf6d830"
        
        # Option 2: Or use built-in site keys for different challenges
        # demo_url = SiteKey.as_site_link(SiteKey.user_easy)     # Easy challenges
        # demo_url = SiteKey.as_site_link(SiteKey.discord)       # Discord challenges  
        # demo_url = SiteKey.as_site_link(SiteKey.epic)          # Epic Games challenges
        # demo_url = SiteKey.as_site_link("a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")  # General demo
        
        print(f"🌐 Navigating to: {demo_url}")
        await page.goto(demo_url)

        try:
            # Run the challenge solver
            agent = await challenge(page)
            
            # Print results if successful
            if agent.cr_list:
                cr: CaptchaResponse = agent.cr_list[-1]
                print("\n✅ Challenge completed successfully!")
                print("📊 Response details:")
                print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))
            else:
                print("❌ No successful challenge response received")
                print("💡 This might be due to:")
                print("   - Model accuracy issues with current challenge type")
                print("   - Network timeouts (try increasing EXECUTION_TIMEOUT)")
                print("   - LLaVA model not suitable for this specific challenge")
                
        except Exception as e:
            print(f"❌ Error during challenge: {e}")
            print("💡 Troubleshooting tips:")
            print("   - Ensure Ollama is running: ollama serve")
            print("   - Check if llava:latest is installed: ollama list")
            print("   - Try a different challenge type")
            print("   - Check the logs for detailed error information")
        
        # Keep browser open for a moment to see results
        print("\n🎉 Demo completed! Browser will close in 10 seconds...")
        print("📝 Check the tmp/.challenge directory for debugging files")
        await asyncio.sleep(10)
        
        await browser.close()


async def test_ollama_connection():
    """Test Ollama connection and model availability before running main demo."""
    print("🔍 Testing Ollama connection...")
    
    try:
        from hcaptcha_challenger.tools.ollama_client import OllamaClient
        
        async with OllamaClient() as client:
            # Test basic connectivity
            response = await client.generate(
                model="llava:latest",
                prompt="Hello, can you see this message? Respond with 'Yes, I can see your message.'",
                options={"temperature": 0.1}
            )
            
            if response and "response" in response:
                print("✅ Ollama connection test successful!")
                print(f"📝 Model response: {response['response'][:100]}...")
                return True
            else:
                print("❌ Ollama responded but with unexpected format")
                return False
                
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
        print("💡 Make sure:")
        print("   1. Ollama is running: ollama serve")
        print("   2. LLaVA model is installed: ollama pull llava:latest")
        print("   3. Ollama is accessible at http://localhost:11434")
        return False


if __name__ == "__main__":
    print("🏁 hCaptcha Challenger - Ollama Edition")
    print("=" * 50)
    
    # First test Ollama connection
    connection_ok = asyncio.run(test_ollama_connection())
    
    if connection_ok:
        print("\n🚀 Starting main demo...")
        asyncio.run(main())
    else:
        print("\n⛔ Ollama connection failed. Please fix the issues above and try again.")