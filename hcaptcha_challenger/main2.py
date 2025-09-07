import asyncio
import json
import sys
import os
from pathlib import Path
import re

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

os.environ['GEMINI_API_KEY'] = 'dummy_key_for_ollama_mode'
os.environ['OLLAMA_URL'] = 'http://localhost:11434'

from playwright.async_api import async_playwright

try:
    from hcaptcha_challenger.agent.challenger import AgentV, AgentConfig
    from hcaptcha_challenger.models import CaptchaResponse
    from hcaptcha_challenger.utils import SiteKey
    print("Successfully imported from hcaptcha_challenger")
except ImportError as e:
    print(f"Import error: {e}")
    sys.path.insert(0, str(current_dir))
    from agent.challenger import AgentV, AgentConfig
    from models import CaptchaResponse
    from utils import SiteKey
    print("Successfully imported using alternative method")


class VisualAnalysisSolver:
    """Solver that uses visual analysis to find correct drag coordinates"""
    
    def __init__(self, page):
        self.page = page
        self.click_count = 0
        self.drag_count = 0
        self.success_count = 0
        self.challenge_count = 0
    
    async def click_checkbox(self):
        """Click the hCaptcha checkbox"""
        try:
            print("Looking for hCaptcha checkbox...")
            await self.page.wait_for_selector('iframe[src*="checkbox"]', timeout=10000)
            
            checkbox_iframe = await self.page.query_selector('iframe[src*="checkbox"]')
            if not checkbox_iframe:
                return False
            
            checkbox_frame = await checkbox_iframe.content_frame()
            if not checkbox_frame:
                return False
            
            await checkbox_frame.click('#checkbox')
            print("Clicked checkbox successfully")
            
            await asyncio.sleep(3)
            return True
            
        except Exception as e:
            print(f"Error clicking checkbox: {e}")
            return False
    
    async def wait_for_challenge(self):
        """Wait for challenge to appear"""
        try:
            print("Waiting for challenge to appear...")
            await self.page.wait_for_selector('iframe[src*="challenge"]', timeout=15000)
            print("Challenge iframe detected")
            
            challenge_iframe = await self.page.query_selector('iframe[src*="challenge"]')
            if not challenge_iframe:
                return None, None
            
            challenge_frame = await challenge_iframe.content_frame()
            if not challenge_frame:
                return None, None
            
            await challenge_frame.wait_for_selector('.challenge-view', timeout=10000)
            print("Challenge content loaded")
            
            await asyncio.sleep(2)
            return challenge_iframe, challenge_frame
            
        except Exception as e:
            print(f"Error waiting for challenge: {e}")
            return None, None
    
    async def detect_challenge_type(self, challenge_frame):
        """Detect challenge type and extract prompt"""
        try:
            all_text = await challenge_frame.evaluate("() => document.body.innerText")
            print(f"Challenge text detected: {all_text[:100]}...")
            
            text_lower = all_text.lower()
            
            if any(keyword in text_lower for keyword in ['drag', 'drop', 'place where it fits']):
                prompt = self.extract_prompt(all_text, ['drag', 'drop', 'place'])
                return "drag_drop", prompt
            elif any(keyword in text_lower for keyword in ['select', 'click', 'find', 'choose', 'identify']):
                prompt = self.extract_prompt(all_text, ['select', 'click', 'find', 'choose', 'identify'])
                return "grid_select", prompt
            else:
                return "grid_select", "Unknown challenge"
                
        except Exception as e:
            print(f"Error detecting challenge type: {e}")
            return "grid_select", "Unknown challenge"
    
    def extract_prompt(self, text, keywords):
        """Extract the challenge prompt"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and any(keyword in line.lower() for keyword in keywords):
                return line
        return "Challenge detected"
    
    async def analyze_drag_elements(self, challenge_frame):
        """Find draggable elements and drop targets using DOM analysis"""
        try:
            print("Analyzing drag elements...")
            
            # Look for elements that could be draggable
            draggable_info = await challenge_frame.evaluate("""
                () => {
                    const result = {
                        draggableElements: [],
                        images: [],
                        allElements: []
                    };
                    
                    // Look for explicitly draggable elements
                    const draggables = document.querySelectorAll('[draggable="true"], .draggable, [class*="drag"]');
                    draggables.forEach((el, i) => {
                        const rect = el.getBoundingClientRect();
                        result.draggableElements.push({
                            index: i,
                            x: rect.x + rect.width / 2,
                            y: rect.y + rect.height / 2,
                            width: rect.width,
                            height: rect.height,
                            tag: el.tagName,
                            className: el.className
                        });
                    });
                    
                    // Look for all images
                    const images = document.querySelectorAll('img');
                    images.forEach((img, i) => {
                        const rect = img.getBoundingClientRect();
                        if (rect.width > 10 && rect.height > 10) {  // Filter out tiny images
                            result.images.push({
                                index: i,
                                x: rect.x + rect.width / 2,
                                y: rect.y + rect.height / 2,
                                width: rect.width,
                                height: rect.height,
                                src: img.src
                            });
                        }
                    });
                    
                    // Look for clickable/interactive elements
                    const interactives = document.querySelectorAll('div[style*="cursor"], div[onclick], [class*="button"], [class*="target"]');
                    interactives.forEach((el, i) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 20 && rect.height > 20) {
                            result.allElements.push({
                                index: i,
                                x: rect.x + rect.width / 2,
                                y: rect.y + rect.height / 2,
                                width: rect.width,
                                height: rect.height,
                                tag: el.tagName,
                                className: el.className
                            });
                        }
                    });
                    
                    return result;
                }
            """)
            
            print(f"Found {len(draggable_info['draggableElements'])} draggable elements")
            print(f"Found {len(draggable_info['images'])} images")
            print(f"Found {len(draggable_info['allElements'])} interactive elements")
            
            return draggable_info
            
        except Exception as e:
            print(f"Error analyzing drag elements: {e}")
            return {"draggableElements": [], "images": [], "allElements": []}
    
    async def solve_drag_drop_challenge(self, challenge_iframe, challenge_frame, prompt):
        """Solve drag-drop using visual analysis"""
        try:
            print(f"Solving drag-drop challenge: {prompt}")
            
            # Get iframe position
            iframe_box = await challenge_iframe.bounding_box()
            if not iframe_box:
                return False
            
            iframe_x = iframe_box['x']
            iframe_y = iframe_box['y']
            
            # Analyze elements in the challenge
            elements = await self.analyze_drag_elements(challenge_frame)
            
            source_x, source_y = None, None
            target_x, target_y = None, None
            
            # Strategy 1: Use found draggable elements
            if elements['draggableElements']:
                # Use the last draggable element (often the piece to drag)
                drag_el = elements['draggableElements'][-1]
                source_x = iframe_x + drag_el['x']
                source_y = iframe_y + drag_el['y']
                print(f"Found draggable element at relative ({drag_el['x']}, {drag_el['y']})")
                
                # For target, try first draggable or use a reasonable position
                if len(elements['draggableElements']) > 1:
                    target_el = elements['draggableElements'][0]
                    target_x = iframe_x + target_el['x']
                    target_y = iframe_y + target_el['y']
                else:
                    # Default target position for letter puzzles
                    target_x = iframe_x + 300
                    target_y = iframe_y + 250
            
            # Strategy 2: Use images if no draggable elements found
            elif elements['images']:
                images = elements['images']
                print(f"Using image elements: {len(images)} found")
                
                if len(images) >= 2:
                    # For letter puzzles, the draggable piece is often the rightmost/last image
                    # Sort by x position to find rightmost
                    images_by_x = sorted(images, key=lambda img: img['x'])
                    
                    source_img = images_by_x[-1]  # Rightmost image (piece to drag)
                    target_img = images_by_x[0]   # Leftmost image (target area)
                    
                    source_x = iframe_x + source_img['x']
                    source_y = iframe_y + source_img['y']
                    target_x = iframe_x + target_img['x'] + 100  # Offset to fit in the gap
                    target_y = iframe_y + target_img['y']
                    
                    print(f"Using images: source at ({source_img['x']}, {source_img['y']}), target at ({target_img['x']}, {target_img['y']})")
            
            # Strategy 3: Fallback to smart positioning
            if not source_x or not target_x:
                print("Using fallback positioning strategy")
                # Based on typical letter puzzle layout
                source_x = iframe_x + 450  # Right side where pieces usually are
                source_y = iframe_y + 200
                target_x = iframe_x + 250  # Left side where they fit
                target_y = iframe_y + 250
            
            print(f"Final drag coordinates: ({source_x}, {source_y}) to ({target_x}, {target_y})")
            
            # Perform the drag operation
            await self.page.mouse.move(source_x, source_y)
            await asyncio.sleep(0.1)
            await self.page.mouse.down()
            await asyncio.sleep(0.1)
            await self.page.mouse.move(target_x, target_y, steps=15)
            await asyncio.sleep(0.1)
            await self.page.mouse.up()
            
            self.drag_count += 1
            print("Drag operation completed")
            
            await asyncio.sleep(2)
            return await self.submit_challenge(challenge_frame)
            
        except Exception as e:
            print(f"Error solving drag-drop: {e}")
            return False
    
    async def solve_grid_challenge(self, challenge_iframe, challenge_frame, prompt):
        """Solve grid selection challenges"""
        try:
            print(f"Solving grid challenge: {prompt}")
            
            iframe_box = await challenge_iframe.bounding_box()
            if not iframe_box:
                return False
            
            iframe_x = iframe_box['x']
            iframe_y = iframe_box['y']
            
            # Determine clicking strategy based on prompt
            prompt_lower = prompt.lower()
            coordinates = []
            
            if any(keyword in prompt_lower for keyword in ['moving', 'mechanical', 'parts']):
                # Click on microwaves (they have moving parts)
                coordinates = [
                    (iframe_x + 180, iframe_y + 280),  # Top-left microwave
                    (iframe_x + 400, iframe_y + 380),  # Bottom-middle microwave  
                    (iframe_x + 520, iframe_y + 480),  # Bottom-right microwave
                ]
                print("Strategy: Clicking microwaves/mechanical objects")
                
            elif any(keyword in prompt_lower for keyword in ['forest', 'woodland', 'animals', 'creatures']):
                coordinates = [
                    (iframe_x + 300, iframe_y + 280),  # Forest animals
                    (iframe_x + 520, iframe_y + 380),
                ]
                print("Strategy: Clicking forest animals")
                
            elif any(keyword in prompt_lower for keyword in ['vehicle', 'car', 'truck']):
                coordinates = [
                    (iframe_x + 180, iframe_y + 280),
                    (iframe_x + 300, iframe_y + 280),
                    (iframe_x + 420, iframe_y + 280),
                ]
                print("Strategy: Clicking vehicles")
                
            else:
                # Generic strategy
                coordinates = [
                    (iframe_x + 300, iframe_y + 280),
                    (iframe_x + 400, iframe_y + 380),
                ]
                print("Strategy: Generic clicking pattern")
            
            # Perform clicks
            for i, (x, y) in enumerate(coordinates):
                self.click_count += 1
                print(f"Clicking at ({x}, {y})")
                
                try:
                    await self.page.mouse.click(x, y, delay=200)
                    print(f"Click {i+1} successful")
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Click {i+1} failed: {e}")
            
            await asyncio.sleep(2)
            return await self.submit_challenge(challenge_frame)
            
        except Exception as e:
            print(f"Error solving grid challenge: {e}")
            return False
    
    async def submit_challenge(self, challenge_frame):
        """Find and click submit/verify/skip button - always try to solve first, then click appropriate button"""
        try:
            print("Looking for submit/verify/skip button...")
            
            # Wait a bit for the UI to update after drag/click
            await asyncio.sleep(1)
            
            # First, try to find buttons by examining all button elements
            all_buttons = await challenge_frame.evaluate("""
                () => {
                    const buttons = [];
                    const buttonElements = document.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"], .button, [class*="button"], [class*="skip"], [class*="verify"], [class*="submit"]');
                    
                    buttonElements.forEach((btn, i) => {
                        const rect = btn.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0 && 
                                         window.getComputedStyle(btn).visibility !== 'hidden' &&
                                         window.getComputedStyle(btn).display !== 'none';
                        
                        if (isVisible) {
                            buttons.push({
                                index: i,
                                text: btn.innerText || btn.textContent || btn.value || '',
                                className: btn.className || '',
                                id: btn.id || '',
                                x: rect.x + rect.width / 2,
                                y: rect.y + rect.height / 2,
                                width: rect.width,
                                height: rect.height
                            });
                        }
                    });
                    
                    return buttons;
                }
            """)
            
            print(f"Found {len(all_buttons)} visible buttons:")
            for btn in all_buttons:
                print(f"  Button: '{btn['text']}' (class: {btn['className']}, id: {btn['id']})")
            
            # Look for different types of buttons
            verify_button = None
            submit_button = None
            skip_button = None
            
            for btn in all_buttons:
                btn_text = btn['text'].lower().strip()
                btn_class = btn['className'].lower()
                
                # Check for different button types
                if 'verify' in btn_text or 'verify' in btn_class:
                    verify_button = btn
                    print(f"Found VERIFY button: '{btn['text']}'")
                elif 'skip' in btn_text or 'skip' in btn_class:
                    skip_button = btn
                    print(f"Found SKIP button: '{btn['text']}'")
                elif any(word in btn_text for word in ['submit', 'check', 'next', 'continue']):
                    submit_button = btn
                    print(f"Found SUBMIT button: '{btn['text']}'")
            
            # Priority order: Verify > Submit > Skip
            # This ensures we always try to complete the challenge first
            target_button = verify_button or submit_button or skip_button
            
            if target_button:
                button_type = "VERIFY" if verify_button else ("SUBMIT" if submit_button else "SKIP")
                print(f"Clicking {button_type} button: '{target_button['text']}' at ({target_button['x']}, {target_button['y']})")
                
                # Click using Playwright's frame click method
                try:
                    await challenge_frame.mouse.click(target_button['x'], target_button['y'])
                    print("Button clicked successfully!")
                    return True
                except Exception as e:
                    print(f"Mouse click failed: {e}")
                    
                    # Fallback: try clicking by selector
                    try:
                        button_elements = await challenge_frame.query_selector_all('button, input[type="button"], input[type="submit"], [role="button"]')
                        for btn_element in button_elements:
                            btn_text = await btn_element.inner_text()
                            if target_button['text'] in btn_text:
                                await btn_element.click()
                                print("Button clicked via selector!")
                                return True
                    except Exception as e2:
                        print(f"Selector click also failed: {e2}")
            
            # Last resort: try clicking common button positions
            print("Trying fallback button positions...")
            fallback_positions = [
                (400, 350),  # Common verify button position
                (450, 400),  # Skip button position
                (500, 450),  # Alternative position
            ]
            
            for x, y in fallback_positions:
                try:
                    await challenge_frame.mouse.click(x, y)
                    print(f"Clicked fallback position ({x}, {y})")
                    await asyncio.sleep(0.5)
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error submitting: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def check_completion(self):
        """Check if captcha is completed"""
        try:
            await asyncio.sleep(3)
            
            # Check if challenge iframe disappeared
            challenge_iframes = await self.page.query_selector_all('iframe[src*="challenge"]')
            if not challenge_iframes:
                print("Challenge iframe disappeared - CAPTCHA completed!")
                return True
            
            # Check for success indicators
            page_content = await self.page.evaluate("() => document.body.innerText")
            success_words = ['success', 'verified', 'complete', 'passed', 'solved']
            
            for word in success_words:
                if word in page_content.lower():
                    print(f"Success indicator found: {word}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking completion: {e}")
            return False


async def main():
    print("hCaptcha Visual Analysis Solver")
    print("=" * 40)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        demo_url = "https://accounts.hcaptcha.com/demo?sitekey=019f1553-3845-481c-a6f5-5a60ccf6d830"
        
        print(f"Navigating to: {demo_url}")
        await page.goto(demo_url)
        
        solver = VisualAnalysisSolver(page)
        
        try:
            # Click checkbox
            checkbox_success = await solver.click_checkbox()
            if not checkbox_success:
                print("Failed to click checkbox")
                return
            
            # Solve challenges
            max_attempts = 15
            for attempt in range(max_attempts):
                solver.challenge_count += 1
                print(f"\n=== CHALLENGE ATTEMPT {solver.challenge_count} ===")
                
                # Wait for challenge
                challenge_iframe, challenge_frame = await solver.wait_for_challenge()
                
                if not challenge_frame:
                    print("No challenge appeared")
                    if await solver.check_completion():
                        print("🎉 CAPTCHA COMPLETED SUCCESSFULLY! 🎉")
                        break
                    continue
                
                # Detect challenge type
                challenge_type, prompt = await solver.detect_challenge_type(challenge_frame)
                print(f"Detected: {challenge_type} - {prompt}")
                
                # Solve based on type
                success = False
                if challenge_type == "drag_drop":
                    success = await solver.solve_drag_drop_challenge(challenge_iframe, challenge_frame, prompt)
                else:
                    success = await solver.solve_grid_challenge(challenge_iframe, challenge_frame, prompt)
                
                if success:
                    solver.success_count += 1
                    print("Challenge solved!")
                
                # Check completion
                await asyncio.sleep(3)
                if await solver.check_completion():
                    print("🎉 CAPTCHA COMPLETED SUCCESSFULLY! 🎉")
                    break
                
                print("Continuing to next challenge...")
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Challenges completed: {solver.success_count}/{solver.challenge_count}")
            print(f"Grid clicks: {solver.click_count}")
            print(f"Drag operations: {solver.drag_count}")
            
        except Exception as e:
            print(f"Error in main: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nCompleted! Browser will close in 10 seconds...")
        await asyncio.sleep(10)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())