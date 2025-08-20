from openai import AzureOpenAI
import os, time, base64
from typing import Optional, List, Dict, Any, Tuple
import PyPDF2
from pathlib import Path
import argparse
import json
import hashlib
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup

# Global verbose flag
VERBOSE = False

@dataclass
class WebsiteConfig:
    """Configuration for a single website with crawling options"""
    url: str
    crawl_depth: int = 0
    max_crawl_pages: int = 5
    name: str = ""  # Optional name for the website
    
    def __post_init__(self):
        if not self.name:
            # Extract a simple name from the URL
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            self.name = parsed.netloc.replace("www.", "").replace(".com", "").replace(".org", "").replace(".edu", "")

@dataclass
class AgentProfile:
    """Individual agent configuration with background and specialty"""
    name: str
    model: str
    specialty: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 10000
    char_limit: int = 2000  # Character limit for background content
    # Individual background sources
    pdfs: List[str] = None
    images: List[str] = None
    websites: List[WebsiteConfig] = None  # Changed from List[str] to List[WebsiteConfig]
    text_files: List[str] = None
    
    def __post_init__(self):
        if self.pdfs is None:
            self.pdfs = []
        if self.images is None:
            self.images = []
        if self.websites is None:
            self.websites = []
        if self.text_files is None:
            self.text_files = []

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_web_context_to_logs(url: str, content: str, agent_name: str = "Unknown") -> str:
    """Save web crawled context to logs/web/ directory"""
    try:
        # Create web logs directory
        web_log_dir = Path("logs/web")
        web_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from URL and timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Clean URL for filename
        safe_url = url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
        safe_url = safe_url[:100]  # Limit length
        filename = f"{timestamp}_{agent_name}_{safe_url}.txt"
        
        # Save content to file
        file_path = web_log_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Agent: {agent_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Content Length: {len(content)} characters\n")
            f.write("="*80 + "\n")
            f.write(content)
        
        if VERBOSE:
            print(f"## Web context saved to: {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        if VERBOSE:
            print(f"## Error saving web context: {str(e)}")
        return ""

def fetch_webpage_content(url: str, agent_name: str = "Unknown", crawl_depth: int = 0, max_pages: int = 5) -> str:
    """Fetch webpage content and optionally crawl linked pages"""
    try:
        if VERBOSE:
            print(f"## Fetching webpage: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # If crawl_depth > 0, follow links and gather additional content
        if crawl_depth > 0 and max_pages > 1:
            if VERBOSE:
                print(f"## Starting crawling: depth={crawl_depth}, max_pages={max_pages}")
            crawled_content = crawl_webpage_links(url, soup, agent_name, crawl_depth, max_pages)
            if crawled_content:
                if VERBOSE:
                    print(f"## Crawling completed, found {len(crawled_content)} characters of linked content")
                text += f"\n\n=== Additional Content from Linked Pages ===\n{crawled_content}"
            else:
                if VERBOSE:
                    print(f"## Crawling completed, no additional content found")
        else:
            if VERBOSE:
                print(f"## No crawling: depth={crawl_depth}, max_pages={max_pages}")
        
        # Save to logs if content is substantial
        if len(text) > 100:  # Only save if there's meaningful content
            save_web_context_to_logs(url, text, agent_name)
        
        return text
        
    except Exception as e:
        if VERBOSE:
            print(f"## Error fetching webpage {url}: {str(e)}")
        return f"Error fetching webpage: {str(e)}"

def crawl_webpage_links(base_url: str, soup: BeautifulSoup, agent_name: str, 
                        remaining_depth: int, remaining_pages: int) -> str:
    """Crawl linked pages to gather additional content"""
    if remaining_depth <= 0 or remaining_pages <= 0:
        return ""
    
    try:
        # Extract the base domain for comparison
        from urllib.parse import urlparse
        base_parsed = urlparse(base_url)
        base_domain = base_parsed.netloc
        
        # Extract links from the page
        links = []
        all_links_found = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            all_links_found.append(href)
            
            # Convert relative URLs to absolute URLs
            if href.startswith('/'):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
            elif href.startswith('#'):
                continue  # Skip anchor links
            elif href.startswith('javascript:'):
                continue  # Skip JavaScript links
            elif href.startswith('mailto:'):
                continue  # Skip email links
            elif href.startswith('//'):
                # Handle protocol-relative URLs
                href = base_parsed.scheme + ':' + href
            
            # Only follow links to the same domain
            try:
                link_parsed = urlparse(href)
                link_domain = link_parsed.netloc
                
                if VERBOSE:
                    print(f"## Checking link: {href} -> domain: {link_domain}")
                
                # Check if it's the same domain or a subdomain
                if link_domain == base_domain or link_domain.endswith('.' + base_domain):
                    # Additional check: make sure it's a content page (not admin, etc.)
                    if not any(skip in href.lower() for skip in ['/admin/', '/login', '/logout', '/edit', '/delete']):
                        links.append(href)
                        if VERBOSE:
                            print(f"## ✅ Added link: {href}")
                    else:
                        if VERBOSE:
                            print(f"## ❌ Skipped admin link: {href}")
                else:
                    if VERBOSE:
                        print(f"## ❌ Different domain: {link_domain} != {base_domain}")
                        
            except Exception as e:
                if VERBOSE:
                    print(f"## Error parsing URL {href}: {e}")
                continue
        
        if VERBOSE:
            print(f"## Total links found: {len(all_links_found)}")
            print(f"## Links after filtering: {len(links)}")
            print(f"## First few all links: {all_links_found[:5]}")
            print(f"## First few filtered links: {links[:5]}")
        
        # Limit the number of links to follow
        links = links[:remaining_pages]  # Follow up to remaining_pages links
        
        if VERBOSE:
            print(f"## Crawling {len(links)} links from {base_url}")
            print(f"## Base domain: {base_domain}")
            for link in links:
                print(f"##   - {link}")
        
        crawled_content = []
        for link_url in links:
            try:
                if VERBOSE:
                    print(f"## Following link: {link_url}")
                
                # Fetch the linked page
                response = requests.get(link_url, timeout=20)
                response.raise_for_status()
                
                # Parse and extract text
                link_soup = BeautifulSoup(response.content, 'html.parser')
                for script in link_soup(["script", "style"]):
                    script.decompose()
                
                link_text = link_soup.get_text()
                lines = (line.strip() for line in link_text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                link_text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Truncate long content
                if len(link_text) > 5000:
                    link_text = link_text[:5000] + "..."
                
                crawled_content.append(f"From {link_url}:\n{link_text}")
                
                if VERBOSE:
                    print(f"## ✅ Successfully crawled {link_url}, content length: {len(link_text)}")
                
                # Recursively crawl if depth allows
                if remaining_depth > 1 and remaining_pages > 1:
                    sub_content = crawl_webpage_links(link_url, link_soup, agent_name, 
                                                   remaining_depth - 1, remaining_pages - 1)
                    if sub_content:
                        crawled_content.append(f"Sub-links from {link_url}:\n{sub_content}")
                
            except Exception as e:
                if VERBOSE:
                    print(f"## Error crawling link {link_url}: {str(e)}")
                continue
        
        if VERBOSE:
            print(f"## Total crawled content pieces: {len(crawled_content)}")
            if crawled_content:
                print(f"## First crawled piece preview: {crawled_content[0][:200]}...")
        
        result = "\n\n".join(crawled_content)
        if VERBOSE:
            print(f"## Final crawled content length: {len(result)}")
        
        return result
        
    except Exception as e:
        if VERBOSE:
            print(f"## Error during crawling: {str(e)}")
        return ""

def crawl_website(url: str, agent_name: str = "Unknown", max_depth: int = 2, max_pages: int = 10) -> str:
    """Crawl a website to gather comprehensive content"""
    try:
        if VERBOSE:
            print(f"## Starting website crawl: {url}")
            print(f"## Max depth: {max_depth}, Max pages: {max_pages}")
        
        # Start crawling from the main page
        content = fetch_webpage_content(url, agent_name, crawl_depth=max_depth, max_pages=max_pages)
        
        if VERBOSE:
            print(f"## Website crawl completed for: {url}")
        
        return content
        
    except Exception as e:
        if VERBOSE:
            print(f"## Error during website crawl: {str(e)}")
        return f"Error crawling website: {str(e)}"

def build_agent_background(agent: AgentProfile) -> str:
    """Build comprehensive background context for an agent"""
    background_parts = []
    
    # Add specialty
    if agent.specialty:
        background_parts.append(f"Specialty: {agent.specialty}")
        background_parts.append("Given your specialty, answer the user's question accurately and concisely.")
    
    
    # Add PDF content
    if agent.pdfs:
        for pdf_path in agent.pdfs:
            if Path(pdf_path).exists():
                try:
                    pdf_text = extract_pdf_text(pdf_path)
                    background_parts.append(f"Relevant PDF Background ({pdf_path}):\n{pdf_text[:agent.char_limit]}...")
                except Exception as e:
                    background_parts.append(f"PDF Error ({pdf_path}): {str(e)}")
    
    # Add image descriptions (simplified)
    if agent.images:
        for img_path in agent.images:
            if Path(img_path).exists():
                background_parts.append(f"RelevantImage Background: {img_path}")
    
    # Add website content
    if agent.websites:
        for website_config in agent.websites:
            if website_config.crawl_depth > 0:
                web_content = crawl_website(website_config.url, website_config.name, website_config.crawl_depth, website_config.max_crawl_pages)
                # Check if crawling found additional content
                if "=== Additional Content from Linked Pages ===" in web_content:
                    background_parts.append(f"Relevant Website Background ({website_config.url}) - WITH CRAWLING (depth={website_config.crawl_depth}, max_pages={website_config.max_crawl_pages}):\n{web_content[:agent.char_limit]}...")
                else:
                    background_parts.append(f"Relevant Website Background ({website_config.url}) - CRAWLING ATTEMPTED BUT NO ADDITIONAL CONTENT:\n{web_content[:agent.char_limit]}...")
            else:
                web_content = fetch_webpage_content(website_config.url, website_config.name)
                background_parts.append(f"Relevant Website Background ({website_config.url}) - SINGLE PAGE:\n{web_content[:agent.char_limit]}...")
    
    # Add text file content
    if agent.text_files:
        for txt_path in agent.text_files:
            if Path(txt_path).exists():
                try:
                    txt_content = read_text_file(txt_path)
                    background_parts.append(f"Relevant Text Background ({txt_path}):\n{txt_content[:agent.char_limit]}...")
                except Exception as e:
                    background_parts.append(f"Text Error ({txt_path}): {str(e)}")
    
    return "\n\n".join(background_parts) if background_parts else "No specific background provided."

def query_gpt(prompt: str, file_path: str = None, conversation_history: list = None, 
              model: str = "gpt-5", temperature: float = 0.7, max_tokens: int = 1000) -> Tuple[str, float]:
    """Query GPT model and return response with timing"""
    
    try:
        # Initialize client based on model type
        client = None
        
        if 'o3' in model:
            client = AzureOpenAI(
                api_key=os.environ.get('AZURE_o3_API_KEY'),
                api_version='2024-12-01-preview', 
                azure_endpoint=os.environ.get('AZURE_o3_API_BASE'),
                azure_deployment=model
            )
        elif 'o1' in model:
            model = 'use-o1'
            client = AzureOpenAI(
                api_key=os.environ.get('AZURE_o1_API_KEY'),
                api_version='2024-12-01-preview', 
                azure_endpoint=os.environ.get('AZURE_o1_API_BASE'),
                azure_deployment=model
            )
        elif 'gpt-5' in model:
            model = 'gpt-5'
            client = AzureOpenAI(
                api_key=os.environ.get('AZURE_o1_API_KEY'),
                api_version='2024-12-01-preview', 
                azure_endpoint=os.environ.get('AZURE_o1_API_BASE'),
                azure_deployment=model
            )
        elif '4o' in model:
            model = 'use4o'
            client = AzureOpenAI(
                api_key=os.environ.get('AZURE_API_KEY'),
                api_version='2024-12-01-preview',  
                azure_endpoint=os.environ.get('AZURE_API_BASE'),
                azure_deployment=model
            )
        else:
            # Default to gpt-5 if model not recognized
            client = AzureOpenAI(
                api_key=os.environ.get('AZURE_o3_API_KEY'),
                api_version='2024-12-01-preview', 
                azure_endpoint=os.environ.get('AZURE_o3_API_BASE'),
                azure_deployment='gpt-5'
            )
        
        if not client:
            raise ValueError(f"Failed to initialize client for model: {model}")
        
        messages = conversation_history if conversation_history else []
        current_message = None
        
        if file_path:
            file_extension = Path(file_path).suffix.lower()
            if file_extension == '.pdf':
                # Handle PDF
                text = extract_pdf_text(file_path)
                current_message = {
                    "role": "user",
                    "content": f"{prompt}\n\nDocument content:\n{text}"
                }
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # Handle image
                base64_image = encode_image(file_path)
                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            elif file_extension in ['.txt', '.py']:
                # Handle text files
                text = read_text_file(file_path)
                current_message = {
                    "role": "user",
                    "content": f"{prompt}\n\nFile content:\n{text}"
                }
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        else:
            current_message = {"role": "user", "content": prompt}
        
        messages.append(current_message)
        api_t0 = time.time()
        
        if VERBOSE:
            # print(f"## query_gpt: Using temperature={temperature}, max_tokens={max_tokens}")
            print(f"## query_gpt: Using max_tokens={max_tokens}")
        
        # For Azure OpenAI, use model_kwargs for max_completion_tokens
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # temperature=temperature,
            max_completion_tokens=max_tokens
            #model_kwargs={"max_completion_tokens": max_tokens}
        )
        inference_time = time.time()-api_t0
        print(f'## {inference_time = :.1f}s')
        
        response_text = response.choices[0].message.content


        return response_text, inference_time

    except Exception as e:
        print(f"\nError in query_gpt: {str(e)}")
        return f"Error: {str(e)}", 0.0

def multi_agent_deliberation(
    agents: List[AgentProfile],
    user_prompt: str,
    interaction_order: List[int],
    rounds: int = 2,
    log_to_file_func: callable = None,
    clean_slate: bool = False
) -> Tuple[str, str, float, List[Tuple[str, float]]]:
    """
    Multi-agent deliberation with specified interaction order and rounds.
    
    Args:
        agents: List of agent profiles
        user_prompt: User's question or prompt
        interaction_order: Order in which agents should respond (1-based indexing)
        rounds: Number of deliberation rounds. Each round = one complete pass through the interaction order
        log_to_file_func: Optional function to log to file
        clean_slate: If True, clear conversation history between rounds (each round starts fresh)
    
    Returns:
        final_answer: The final collaborative answer
        transcript: Detailed transcript of all interactions
        total_time: Total time taken for deliberation
        step_timings: Timing for each step
    """
    
    if VERBOSE:
        print(f"\n🔄 Starting multi-agent deliberation")
        print(f"   Agents: {len(agents)}")
        print(f"   Interaction Order: {interaction_order}")
        print(f"   Rounds: {rounds}")
    
    # Initialize tracking variables
    agent_responses = [""] * len(agents)  # Store responses for each agent
    agent_histories = [[] for _ in agents]  # Conversation history for each agent
    transcript_lines = []
    step_timings = []
    total_time = 0.0
    
    # Build comprehensive background for each agent and initialize conversation histories
    for i, agent in enumerate(agents):
        background = build_agent_background(agent)
        system_message = f"""You are {agent.name}, a specialized AI agent.

{agent.system_prompt}

Your background knowledge includes:
{background}

You are now participating in a multi-agent deliberation to answer: {user_prompt}

Remember your specialty and background when contributing to the discussion."""
        
        agent_histories[i] = [{"role": "system", "content": system_message}]
    
    # Log deliberation start
    if log_to_file_func:
        log_to_file_func(f"\n=== Multi-Agent Deliberation Started ===")
        log_to_file_func(f"User Prompt: {user_prompt}")
        log_to_file_func(f"Interaction Order: {interaction_order}")
        log_to_file_func(f"Rounds: {rounds}")
        log_to_file_func(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process each round
    for round_num in range(rounds):
        if VERBOSE:
            print(f"\n🔄 Round {round_num + 1}/{rounds}")
            print(f"   Processing agents in order: {interaction_order}")
            if clean_slate and round_num > 0:
                print(f"   🧹 Clean slate mode: Starting fresh (no previous round context)")
        
        # Log round start
        if log_to_file_func:
            log_to_file_func(f"\n--- Round {round_num + 1} Started ---")
            log_to_file_func(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if clean_slate and round_num > 0:
                log_to_file_func(f"Clean slate mode: Starting fresh (no previous round context)")
        
        # If clean slate is enabled and this is not the first round, reset agent responses
        if clean_slate and round_num > 0:
            agent_responses = [""] * len(agents)
            if VERBOSE:
                print(f"   🧹 Clean slate: Cleared all previous agent responses")
            if log_to_file_func:
                log_to_file_func(f"  Clean slate: Cleared all previous agent responses")
        
        # Process each agent in the specified order for this round
        for agent_order_idx, agent_num in enumerate(interaction_order):
            agent_idx = agent_num - 1  # Convert to 0-based index
            agent = agents[agent_idx]
            
            if VERBOSE:
                print(f"   🤖 {agent.name} (Agent #{agent_num}) responding...")
            
            # Log agent start
            if log_to_file_func:
                log_to_file_func(f"  {agent.name} (Round {round_num + 1}, Agent #{agent_num}) starting...")
            
            # Build context from other agents' responses (only if not clean slate or first round)
            other_responses = []
            if not clean_slate or round_num == 0:
                for other_idx, other_response in enumerate(agent_responses):
                    if other_idx != agent_idx and other_response:
                        other_responses.append(f"{agents[other_idx].name}: {other_response}")
            
            # Create prompt for this agent
            if other_responses:
                prompt = f"""User Question: {user_prompt}

Previous responses from other agents:
{chr(10).join(other_responses)}

Based on your specialty and background, provide an improved response that builds upon the discussion."""
            else:
                prompt = f"""User Question: {user_prompt}

Based on your specialty and background, provide your response."""
            
            # Query the agent
            start_time = time.time()
            response = query_gpt(
                prompt, 
                conversation_history=agent_histories[agent_idx], 
                model=agent.model,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens
            )
            
            if response is None:
                if VERBOSE:
                    print(f"## {agent.name} failed to respond")
                transcript_lines.append(f"### {agent.name} (Round {round_num + 1}): [FAILED]")
                if log_to_file_func:
                    log_to_file_func(f"  {agent.name} (Round {round_num + 1}): [FAILED]")
                continue
            
            response_text, inference_time = response
            step_time = time.time() - start_time
            total_time += step_time
            step_timings.append((f"{agent.name} (Round {round_num + 1})", step_time))
            
            # Update agent response
            agent_responses[agent_idx] = response_text
            
            # Add to transcript
            transcript_lines.append(f"### {agent.name} (Round {round_num + 1}): {response_text}")
            
            if VERBOSE:
                print(f"## {agent.name} response: {response_text[:100]}...")
            
            # Log individual agent response to file
            if log_to_file_func:
                log_to_file_func(f"  {agent.name} (Round {round_num + 1}): {response_text}")
        
        # Round summary with detailed agent responses
        if VERBOSE:
            print(f"\n## Round {round_num + 1} Complete - {len(interaction_order)} agents processed")
            print(f"## Round {round_num + 1} Agent Responses:")
            for i, agent_idx in enumerate(interaction_order):
                agent_idx = agent_idx - 1  # Convert to 0-based index
                agent_name = agents[agent_idx].name
                response = agent_responses[agent_idx]
                if response:
                    print(f"   {agent_name}: {response[:100]}...")
                else:
                    print(f"   {agent_name}: [No response]")
        
        # Log detailed round summary to file
        if log_to_file_func:
            log_to_file_func(f"\n=== Round {round_num + 1} Complete ===")
            log_to_file_func(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            log_to_file_func(f"Agents Processed: {len(interaction_order)}")
            for i, agent_idx in enumerate(interaction_order):
                agent_idx = agent_idx - 1  # Convert to 0-based index
                agent_name = agents[agent_idx].name
                response = agent_responses[agent_idx]
                if response:
                    log_to_file_func(f"  {agent_name}: {response}")
                else:
                    log_to_file_func(f"  {agent.name}: [No response]")
    
    # Determine final answer (use the last agent in the final round)
    final_answer = ""
    if agent_responses:
        final_agent_idx = interaction_order[-1] - 1
        final_answer = agent_responses[final_agent_idx]
        
        # If that failed, use the first successful response
        if not final_answer:
            for response in agent_responses:
                if response:
                    final_answer = response
                    break
    
    if not final_answer:
        final_answer = "No agents provided a valid response."
    
    # Create comprehensive response summary
    response_summary = []
    response_summary.append("=== COMPREHENSIVE AGENT RESPONSES ===")
    response_summary.append(f"Total Rounds: {rounds}")
    response_summary.append(f"Interaction Order: {interaction_order}")
    response_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for round_num in range(rounds):
        response_summary.append(f"\n--- Round {round_num + 1} ---")
        for i, agent_idx in enumerate(interaction_order):
            agent_idx = agent_idx - 1  # Convert to 0-based index
            agent_name = agents[agent_idx].name
            response = agent_responses[agent_idx]
            if response:
                response_summary.append(f"  {agent_name}: {response}")
            else:
                response_summary.append(f"  {agent_name}: [No response]")
    
    response_summary.append(f"\n=== FINAL ANSWER ===")
    response_summary.append(f"{final_answer}")
    
    # Create transcript from transcript lines
    transcript = "\n".join(transcript_lines)
    
    # Add response summary to transcript
    detailed_transcript = transcript + "\n\n" + "\n".join(response_summary)
    
    # Log final summary
    if log_to_file_func:
        log_to_file_func(f"\n=== Deliberation Complete ===")
        log_to_file_func(f"Final Answer: {final_answer}")
        log_to_file_func(f"Total Time: {total_time:.2f}s")
        log_to_file_func(f"Step Timings:")
        for step, timing in step_timings:
            log_to_file_func(f"  - {step}: {timing:.2f}s")
    
    return final_answer, detailed_transcript, total_time, step_timings

def create_agent_profiles_from_config(config: dict) -> List[AgentProfile]:
    """Create agent profiles from configuration dictionary"""
    agents = []
    
    # Get global defaults from config
    global_model = config.get('model', 'gpt-5')
    global_temperature = config.get('temperature', 0.7)
    global_max_tokens = config.get('max_tokens', 10000)
    
    for agent_config in config.get('agents', []):
        # Convert website strings to WebsiteConfig objects
        websites = []
        for website in agent_config.get('websites', []):
            if isinstance(website, str):
                # Backward compatibility: convert string to WebsiteConfig
                websites.append(WebsiteConfig(url=website))
            elif isinstance(website, dict):
                # New format: website with crawling options
                websites.append(WebsiteConfig(
                    url=website.get('url', ''),
                    crawl_depth=website.get('crawl_depth', 0),
                    max_crawl_pages=website.get('max_crawl_pages', 5),
                    name=website.get('name', '')
                ))
        
        agent = AgentProfile(
            name=agent_config.get('name', 'Unknown Agent'),
            model=agent_config.get('model', global_model),  # Use global model as default
            specialty=agent_config.get('specialty', 'General purpose'),
            system_prompt=agent_config.get('system_prompt', 'You are a helpful AI assistant.'),
            temperature=agent_config.get('temperature', global_temperature),  # Use global temperature as default
            max_tokens=agent_config.get('max_tokens', global_max_tokens),  # Use global max_tokens as default
            char_limit=agent_config.get('char_limit', 2000), # Use char_limit from config
            pdfs=agent_config.get('pdfs', []),
            images=agent_config.get('images', []),
            websites=websites,
            text_files=agent_config.get('text_files', [])
        )
        agents.append(agent)
    
    return agents

def interactive_multi_agent_chat(
    agents: List[AgentProfile],
    interaction_order: List[int],
    rounds: int = 2,
    save_log: bool = False,
    default_question: str = None,
    config: Dict[str, Any] = None,
    clean_slate: bool = False
):
    """Interactive chat interface for multi-agent system"""
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create session log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_log_file = log_dir / f"session_{timestamp}.txt"
    
    def log_to_file(content: str, file_path: Path = None):
        """Helper function to log content to file"""
        target_file = file_path or session_log_file
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    
    # Log session start
    log_to_file(f"=== Multi-Agent Chat Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_to_file(f"Timestamp: {timestamp}")
    log_to_file(f"Agents: {len(agents)}")
    log_to_file(f"Interaction Order: {interaction_order}")
    log_to_file(f"Rounds: {rounds}")
    log_to_file(f"Logging: {'Enabled' if save_log else 'Disabled'}")
    
    # Log configuration if provided
    if config:
        log_to_file(f"\n=== Configuration ===")
        log_to_file(json.dumps(config, indent=2, ensure_ascii=False))
        
        # Log global settings separately for clarity
        global_model = config.get('model', 'gpt-5')
        global_temperature = config.get('temperature', 0.7)
        global_max_tokens = config.get('max_tokens', 10000)
        log_to_file(f"\n=== Global Settings ===")
        log_to_file(f"Global Model: {global_model}")
        log_to_file(f"Global Temperature: {global_temperature}")
        log_to_file(f"Global Max Tokens: {global_max_tokens}")
    
    # Log agent profiles
    log_to_file(f"\n=== Agent Profiles ===")
    for i, agent in enumerate(agents):
        log_to_file(f"• Agent {i+1}: {agent.name}")
        log_to_file(f"  🤖 Model: {agent.model}")
        log_to_file(f"  🎯 Specialty: {agent.specialty}")
        log_to_file(f"  💬 Max Tokens: {agent.max_tokens}")
        log_to_file(f"  📄 Char Limit: {agent.char_limit}")
        if agent.websites:
            log_to_file(f"  Websites:")
            for website in agent.websites:
                if website.crawl_depth > 0:
                    log_to_file(f"    - {website.name} ({website.url}) - Crawling: Depth {website.crawl_depth}, Max {website.max_crawl_pages} pages")
                else:
                    log_to_file(f"    - {website.name} ({website.url}) - No crawling")
        log_to_file(f"  PDFs: {agent.pdfs}")
        log_to_file(f"  Images: {agent.images}")
        log_to_file(f"  Text Files: {agent.text_files}")
    
    print(f"\n🤖 Multi-Agent Chat System")
    print(f"📋 {len(agents)} agents configured")
    print(f"🔄 Interaction order: {interaction_order}")
    print(f"🔄 Rounds: {rounds}")
    print(f"💾 Logging: {'Enabled' if save_log else 'Disabled'}")
    print(f"📁 Session log: {session_log_file}")
    
    # Display global settings if available
    if config:
        global_model = config.get('model', 'gpt-5')
        global_temperature = config.get('temperature', 0.7)
        global_max_tokens = config.get('max_tokens', 10000)
        print(f"⚙️  Global Settings:")
        print(f"   Model: {global_model}")
        # print(f"   Temperature: {global_temperature}")
        print(f"   Max Tokens: {global_max_tokens}")
    
    # Display agent information
    print(f"\n Agent Profiles:")
    for i, agent in enumerate(agents):
        print(f"  •  {agent.name} ({agent.model})")
        print(f"     🎯 Specialty: {agent.specialty}")
        print(f"     💬 Max Tokens: {agent.max_tokens}")
        print(f"     📄 Char Limit: {agent.char_limit}")
        if agent.websites:
            print(f"     Websites:")
            for website in agent.websites:
                if website.crawl_depth > 0:
                    print(f"       - {website.name} ({website.url}) - Crawling: Depth {website.crawl_depth}, Max {website.max_crawl_pages} pages")
                else:
                    print(f"       - {website.name} ({website.url}) - No crawling")
        print(f"     📄 PDF: {len(agent.pdfs)} \n     🖼️  images: {len(agent.images)} \n     📚 websites: {len(agent.websites)} \n     📝 text files: {len(agent.text_files)}")
    
    print(f"\n💬 Type your questions or 'exit' to quit")
    print(f"📁 You can also use: pdf: path/to/file.pdf, image: path/to/image.jpg, url: https://example.com")
    
    # If there's a default question, run it automatically
    if default_question:
        print(f"\n🚀 Running default question: {default_question}")
        print(f"🔄 Processing with {len(agents)} agents...")
        
        # Log default question
        log_to_file(f"\n=== Default Question ===")
        log_to_file(f"Question: {default_question}")
        log_to_file(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run multi-agent deliberation for default question
            final_answer, transcript, total_time, step_timings = multi_agent_deliberation(
                agents=agents,
                user_prompt=default_question,
                interaction_order=interaction_order,
                rounds=rounds,
                log_to_file_func=log_to_file, # Pass the log_to_file function
                clean_slate=clean_slate # Pass clean_slate setting
            )
            
            # Log results
            log_to_file(f"\n=== Default Question Results ===")
            log_to_file(f"Final Answer: {final_answer}")
            log_to_file(f"Total Time: {total_time:.2f}s")
            log_to_file(f"Step Timings:")
            for step, timing in step_timings:
                log_to_file(f"  - {step}: {timing:.2f}s")
            log_to_file(f"\nFull Transcript:\n{transcript}")
            
            # Display results
            print(f"\n🤖 Final Collaborative Answer:")
            print(f"{final_answer}")
            
            print(f"\n⏱️  Total time: {total_time:.2f}s")
            print(f"📊 Step timings:")
            for step, timing in step_timings:
                print(f"  - {step}: {timing:.2f}s")
            
            # Save detailed log if enabled
            if save_log:
                detailed_log_file = log_dir / f"default_question_{timestamp}.txt"
                with open(detailed_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Default Question Session: {timestamp}\n")
                    f.write(f"Question: {default_question}\n")
                    f.write(f"Final Answer: {final_answer}\n")
                    f.write(f"Total Time: {total_time:.2f}s\n")
                    f.write(f"Step Timings:\n")
                    for step, timing in step_timings:
                        f.write(f"  - {step}: {timing:.2f}s\n")
                    f.write(f"\nFull Transcript:\n{transcript}\n")
                
                print(f"💾 Detailed log saved to: {detailed_log_file}")
            
            print(f"\n" + "="*60)
            print(f"💬 Now you can ask your own questions or type 'exit' to quit")
            print(f"="*60)
            
        except Exception as e:
            error_msg = f"❌ Error processing default question: {str(e)}"
            print(error_msg)
            log_to_file(f"\n=== Error ===")
            log_to_file(error_msg)
            if VERBOSE:
                import traceback
                traceback.print_exc()
                log_to_file(f"Traceback:\n{traceback.format_exc()}")
    
    conversation_history = []
    question_count = 0
    
    while True:
        try:
            user_input = input("\n## You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                log_to_file(f"\n=== Session Ended ===")
                log_to_file(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                log_to_file(f"Total Questions: {question_count}")
                break
            
            if not user_input:
                continue
            
            question_count += 1
            print(f"\n🔄 Processing Question #{question_count} with {len(agents)} agents...")
            
            # Log user question
            log_to_file(f"\n=== Question #{question_count} ===")
            log_to_file(f"User Input: {user_input}")
            log_to_file(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run multi-agent deliberation
            final_answer, transcript, total_time, step_timings = multi_agent_deliberation(
                agents=agents,
                user_prompt=user_input,
                interaction_order=interaction_order,
                rounds=rounds,
                log_to_file_func=log_to_file, # Pass the log_to_file function
                clean_slate=clean_slate # Pass clean_slate setting
            )
            
            # Log results
            log_to_file(f"\n=== Question #{question_count} Results ===")
            log_to_file(f"Final Answer: {final_answer}")
            log_to_file(f"Total Time: {total_time:.2f}s")
            log_to_file(f"Step Timings:")
            for step, timing in step_timings:
                log_to_file(f"  - {step}: {timing:.2f}s")
            log_to_file(f"\nFull Transcript:\n{transcript}")
            
            # Display results
            print(f"\n🤖 Final Collaborative Answer:")
            print(f"{final_answer}")
            
            print(f"\n⏱️  Total time: {total_time:.2f}s")
            print(f"📊 Step timings:")
            for step, timing in step_timings:
                print(f"  - {step}: {timing:.2f}s")
            
            # Save detailed log if enabled
            if save_log:
                detailed_log_file = log_dir / f"question_{question_count}_{timestamp}.txt"
                with open(detailed_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Question #{question_count} Session: {timestamp}\n")
                    f.write(f"User Input: {user_input}\n")
                    f.write(f"Final Answer: {final_answer}\n")
                    f.write(f"Total Time: {total_time:.2f}s\n")
                    f.write(f"Step Timings:\n")
                    for step, timing in step_timings:
                        f.write(f"  - {step}: {timing:.2f}s\n")
                    f.write(f"\nFull Transcript:\n{transcript}\n")
                
                print(f"💾 Detailed log saved to: {detailed_log_file}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": final_answer})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            log_to_file(f"\n=== Session Interrupted ===")
            log_to_file(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            log_to_file(f"Total Questions: {question_count}")
            break
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            log_to_file(f"\n=== Error in Question #{question_count} ===")
            log_to_file(error_msg)
            if VERBOSE:
                import traceback
                traceback.print_exc()
                log_to_file(f"Traceback:\n{traceback.format_exc()}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Chat System with Individual Backgrounds')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file with agent profiles')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debugging output')
    # parser.add_argument('--rounds', type=int, default=2, help='Number of deliberation rounds (default: 2)')
    parser.add_argument('--save-log', action='store_true', help='Save conversation logs')
    parser.add_argument('--interaction-order', type=str, help='Custom interaction order (e.g., "1,3,2,4")')
    parser.add_argument('--clean-slate', action='store_true', help='Clear conversation history between rounds (each round starts fresh)')
    
    args = parser.parse_args()
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return
    
    if VERBOSE:
        print(f"## Loaded config from: {config_path}")
        print(f"## Config: {json.dumps(config, indent=2)}")
    
    # Create agent profiles
    try:
        agents = create_agent_profiles_from_config(config)
        if not agents:
            print("❌ No agents found in configuration")
            return
        
        print(f"✅ Created {len(agents)} agent profiles")
        
    except Exception as e:
        print(f"❌ Error creating agent profiles: {str(e)}")
        if VERBOSE:
            import traceback
            traceback.print_exc()
        return
    
    # Get interaction order
    if args.interaction_order:
        try:
            interaction_order = [int(x.strip()) for x in args.interaction_order.split(',')]
        except ValueError:
            print("❌ Invalid interaction order format. Use comma-separated numbers (e.g., '1,3,2,4')")
            return
    else:
        interaction_order = config.get('interaction_order', list(range(1, len(agents) + 1)))
    
    # Validate interaction order
    if not interaction_order or max(interaction_order) > len(agents):
        print(f"❌ Invalid interaction order {interaction_order} for {len(agents)} agents")
        print(f"   Valid range: 1 to {len(agents)}")
        return
    
    # Get rounds
    rounds = config.get('rounds', 2)
    
    # Get default question from config
    default_question = config.get('default_question', None)
    
    # Get clean slate setting (CLI argument overrides config)
    clean_slate = args.clean_slate or config.get('clean_slate', False)
    
    if VERBOSE:
        print(f"## Clean slate mode: {'Enabled' if clean_slate else 'Disabled'}")
    
    # Start interactive chat
    try:
        interactive_multi_agent_chat(
            agents=agents,
            interaction_order=interaction_order,
            rounds=rounds,
            save_log=args.save_log,
            default_question=default_question,
            config=config, # Pass config to the interactive function
            clean_slate=clean_slate # Pass clean slate setting
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error in interactive chat: {str(e)}")
        if VERBOSE:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()