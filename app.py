# Tamil Nadu Tourism Guide Chatbot
# Features:
# - Uses Groq LLM with Llama 3.1 model for natural language processing
# - TavilySearchResults for enhanced search capabilities about Tamil Nadu
# - Wikipedia and DuckDuckGo as backup search tools
# - ElevenLabs for high-quality text-to-speech
# - AssemblyAI for speech recognition
# - Improved error handling for questions outside the chatbot's knowledge domain

# Standard library imports
import re
import os
import base64
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import requests
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

# LangChain and AI imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
import assemblyai as aai
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ===============================================================
# API KEY CONFIGURATION
# ===============================================================

# Get API keys from environment variables with fallbacks
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your-groq-api-key')
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY', 'your-elevenlabs-key')
ASSEMBLY_AI_API_KEY = os.getenv('ASSEMBLY_AI_API_KEY', 'your-assemblyai-key')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', 'your-tavily-api-key')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key')

# Print status of API keys
if GROQ_API_KEY == 'your-groq-api-key':
    print("WARNING: Groq API key not set in .env file")
else:
    print("Groq API key loaded successfully")

if ELEVEN_LABS_API_KEY == 'your-elevenlabs-key':
    print("WARNING: ElevenLabs API key not set in .env file")
else:
    print("ElevenLabs API key loaded successfully")

if ASSEMBLY_AI_API_KEY == 'your-assemblyai-key':
    print("WARNING: AssemblyAI API key not set in .env file")
else:
    print("AssemblyAI API key loaded successfully")

if TAVILY_API_KEY == 'your-tavily-api-key':
    print("WARNING: Tavily API key not set in .env file")
else:
    print("Tavily API key loaded successfully")

# Print status of Gemini API key
if GEMINI_API_KEY == 'your-gemini-api-key':
    print("WARNING: Gemini API key not set in .env file")
else:
    print("Gemini API key loaded successfully")

# Configure Gemini
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully")
else:
    print("WARNING: Gemini API not configured due to missing API key")

# ===============================================================
# TOOLS CONFIGURATION
# ===============================================================

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure Tavily Search tool (primary search tool)
if TAVILY_API_KEY and TAVILY_API_KEY != "your-tavily-api-key":
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    tavily_search = TavilySearchResults(
        max_results=3,
        search_depth="advanced",  # Use advanced search for more detailed results
        include_domains=["wikipedia.org", "tamilnadutourism.gov.in", "incredibleindia.org", "lonelyplanet.com", "tripadvisor.in"],
        include_raw_content=True,
        api_key=TAVILY_API_KEY
    )
    tavily_tool = Tool(
        name="tavily_search",
        func=tavily_search.run,
        description="Search for detailed and up-to-date information about Tamil Nadu tourism, attractions, culture, and travel tips. This tool provides high-quality search results focused on Tamil Nadu."
    )
    print("Tavily Search configured successfully")
else:
    tavily_tool = None
    print("WARNING: Tavily Search not configured due to missing API key")

# Configure DuckDuckGo search tool (backup search)
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for general information when specific Tamil Nadu tourism details are not available through other tools",
)

# Configure Wikipedia tool
api_wrapper = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=1000,
)
wiki_tool = WikipediaQueryRun(
    name="wiki_tool",
    api_wrapper=api_wrapper,
)

# ===============================================================
# API CONFIGURATION
# ===============================================================

# Configure Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Configure ElevenLabs for text-to-speech
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Initialize ElevenLabs client
if ELEVEN_LABS_API_KEY and ELEVEN_LABS_API_KEY != "your-elevenlabs-key":
    client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)
    print("ElevenLabs client initialized successfully")
else:
    client = None
    print("WARNING: ElevenLabs client not initialized due to missing API key")

# Configure AssemblyAI for speech recognition
if ASSEMBLY_AI_API_KEY and ASSEMBLY_AI_API_KEY != "your-assemblyai-key":
    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    print("AssemblyAI configured")
else:
    print("WARNING: AssemblyAI key not set. Speech recognition will use browser-based recognition.")

# ===============================================================
# SYSTEM PROMPT CONFIGURATION
# ===============================================================

# Create a more conversational prompt with detailed information and Markdown formatting instructions
prompt = ChatPromptTemplate.from_messages([
    (
        "system", """You are Thamizha, a friendly and knowledgeable Tamil Nadu tourism guide.

Speak in a warm, conversational tone as if chatting with a friend.

IMPORTANT: FORMAT ALL RESPONSES USING MARKDOWN:
• Start with "Vanakkam!" (Hello in Tamil)
• Use ## for main section headings (e.g., ## Famous Temples)
• Use ### for sub-headings
• Use **bold** for important information like place names, prices, timings
• Use bullet points (- ) for listing attractions, features, etc.
• Use numbered lists (1. 2. 3.) for step-by-step instructions or ranked items
• Use horizontal rules (---) between major sections
• Structure your response with clear sections and proper spacing
• Keep paragraphs short (3-4 sentences maximum)

CONTENT GUIDELINES:
• Focus on Tamil Nadu tourism information
• Include specific details about attractions, culture, food, festivals, etc.
• Mention practical information like timings, prices, best times to visit
• Recommend authentic local experiences
• Be enthusiastic and showcase Tamil Nadu's unique features
• Personalize responses based on the traveler's interests when possible

SEARCH RESULT PRESENTATION:
• NEVER show raw error messages like "No good Wikipedia Search Result was found"
• If search fails, respond as if you already know the answer from your knowledge
• For Wikipedia results: "From Wikipedia: [information]" or "Wikipedia mentions that [information]"
• For web search results: "From web search: [information]" or "According to online sources: [information]"
• For Tavily search results: "Based on the latest information: [information]" or "According to current travel guides: [information]"
• Always present search information in a natural, conversational way
• Combine information from multiple sources when appropriate to provide comprehensive answers

EXAMPLE RESPONSE FORMAT:
Vanakkam! I'm delighted to help you explore Tamil Nadu.

## Famous Temples
- **Meenakshi Amman Temple** in Madurai features stunning architecture with thousands of colorful sculptures. Open **5:00 AM to 12:30 PM** and **4:00 PM to 10:00 PM**.
- **Brihadeeswarar Temple** in Thanjavur is a UNESCO World Heritage site built by Raja Raja Chola I.

---

## Beautiful Hill Stations
- **Ooty** (Udhagamandalam) offers lush tea plantations and the famous Nilgiri Mountain Railway. Best visited between **March and June**.
- **Kodaikanal** features the stunning **Coaker's Walk** with panoramic valley views.

Remember to be helpful, accurate, and make the traveler excited about visiting Tamil Nadu!"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Set up tools and agent
tools = []
if tavily_tool:
    tools.append(tavily_tool)  # Add Tavily search as primary tool if available
tools.extend([search_tool, wiki_tool])  # Add DuckDuckGo and Wikipedia tools

# Create agent with better handling of function calls
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=3,  # Limit iterations to prevent excessive API calls
    early_stopping_method="generate"  # Stop early if we get a good response
)

# Add a post-processing function to clean up responses
def clean_response(text):
    """Clean the response text and use Gemini to fix problematic responses.
    
    Args:
        text (str): The raw response text
        
    Returns:
        str: Cleaned response text
    """
    # Remove function call syntax like /function=wiki_tool>{"query": "..."}
    import re
    
    # More comprehensive pattern to catch different function call formats
    cleaned_text = re.sub(r'(?:\/|<)?function=\w+>(?:\{.*?\})+;?', '', text)
    
    # Remove any JSON-like structures that might be function calls
    cleaned_text = re.sub(r'\{\s*"query"\s*:\s*"[^"]*"\s*\}\}?', '', cleaned_text)
    
    # Remove any trailing semicolons from function calls
    cleaned_text = re.sub(r';\s*(\n|$)', r'\1', cleaned_text)
    
    # Remove any double newlines created by removing function calls
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    
    # Check for error patterns or too short responses
    error_patterns = [
        r"No good Wikipedia Search Result was found",
        r"I couldn't find specific information",
        r"I don't have access to",
        r"I don't have information",
        r"I don't have the ability to",
        r"I cannot search",
        r"I cannot browse",
        r"I cannot access",
        r"I'm an AI assistant",
        r"As an AI",
        r"As a language model",
        r"I'm not able to",
        r"I'm unable to",
        r"I don't have the capability",
        r"I don't have current information"
    ]
    
    has_error = any(re.search(pattern, cleaned_text, re.IGNORECASE) for pattern in error_patterns)
    is_too_short = len(cleaned_text.strip()) < 50
    
    # If there's an error or response is too short, use Gemini to generate a better response
    if has_error or is_too_short:
        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
                # Extract any useful information from the original response
                model = genai.GenerativeModel('gemini-pro')
                
                # Create a prompt for Gemini to generate a better response
                prompt = """You are Thamizha, a friendly Tamil Nadu tourism guide.

The following response has issues (it's too short, contains error messages, or doesn't provide helpful information):

"""
                prompt += cleaned_text
                
                prompt += """

Please generate a better response that:
1. Starts with 'Vanakkam!' and is enthusiastic about Tamil Nadu
2. Provides helpful information about Tamil Nadu tourism
3. Includes specific places, attractions, or cultural aspects
4. Is conversational and engaging
5. Doesn't mention being an AI or having limitations

FORMAT YOUR RESPONSE USING MARKDOWN:
• Use ## for main section headings (e.g., ## Famous Temples)
• Use ### for sub-headings
• Use **bold** for important information like place names, prices, timings
• Use bullet points (- ) for listing attractions, features, etc.
• Use numbered lists (1. 2. 3.) for step-by-step instructions or ranked items
• Use horizontal rules (---) between major sections
• Structure your response with clear sections and proper spacing
• Keep paragraphs short (3-4 sentences maximum)

Your response should be as if you're a knowledgeable tour guide speaking directly to a tourist."""
                
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
        except Exception as e:
            print(f"Gemini API error in clean_response: {str(e)}")
    
    # Remove any references to search tools
    cleaned_text = re.sub(r"(From Wikipedia:|According to Wikipedia:|Based on Wikipedia:|From web search:|According to web search:|Based on web search:|From search:|According to search:|Based on search:)", "", cleaned_text)
    
    # Ensure the response starts with a greeting
    if not re.match(r"(Vanakkam|Hello|Hi|Greetings|Welcome)", cleaned_text, re.IGNORECASE):
        cleaned_text = "Vanakkam! " + cleaned_text
    
    return cleaned_text

# Fallback function for when agent execution fails
def fallback_response(query):
    """Provide a fallback response when the agent execution fails.
    
    Attempts to use available search tools in sequence to find relevant information.
    If all search attempts fail, uses Gemini API as a fallback.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: A response based on available information sources
    """
    search_results = ""
    
    try:
        # First try Tavily search if available (best quality results)
        if tavily_tool:
            tavily_result = tavily_tool.func(f"Tamil Nadu tourism {query}")
            if tavily_result and len(tavily_result) > 10:
                search_results += f"Tavily search results: {tavily_result}\n\n"
    except Exception as e:
        print(f"Tavily search error: {str(e)}")

    try:
        # Then try to get information from Wikipedia
        wiki_result = wiki_tool.run(f"Tamil Nadu {query}")
        if wiki_result and len(wiki_result) > 10:
            search_results += f"Wikipedia results: {wiki_result}\n\n"
    except Exception as e:
        print(f"Wikipedia search error: {str(e)}")

    try:
        # Finally try DuckDuckGo search
        search_result = search_tool.run(f"Tamil Nadu tourism {query}")
        if search_result and len(search_result) > 10:
            search_results += f"Web search results: {search_result}\n\n"
    except Exception as e:
        print(f"DuckDuckGo search error: {str(e)}")

    # If we have search results, use Gemini to generate a response based on them
    if search_results:
        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
                # Use Gemini to generate a response based on search results
                model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"""You are Thamizha, a friendly Tamil Nadu tourism guide.
                
Based on the following search results, provide a helpful response to the user's query about Tamil Nadu tourism.
Start with 'Vanakkam!' and be enthusiastic about Tamil Nadu.

USER QUERY: {query}

SEARCH RESULTS:
{search_results}

Your response should be informative, conversational, and focused on Tamil Nadu tourism. 
If the search results don't contain enough information, use your knowledge about Tamil Nadu to provide a helpful response.
Don't mention that you're using search results or that you're an AI. Just provide the information as a knowledgeable tour guide would.
"""
                
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
        except Exception as e:
            print(f"Gemini API error with search results: {str(e)}")
    
    # If no search results or Gemini with search results failed, try direct Gemini query
    try:
        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""You are Thamizha, a friendly Tamil Nadu tourism guide.
            
Provide a helpful response to the following query about Tamil Nadu tourism:
"{query}"

Start with 'Vanakkam!' and be enthusiastic about Tamil Nadu.
Your response should be informative, conversational, and focused on Tamil Nadu tourism.
Include specific places, attractions, or cultural aspects relevant to the query.
Don't mention that you're an AI. Just provide the information as a knowledgeable tour guide would.
"""
            
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
    except Exception as e:
        print(f"Direct Gemini API error: {str(e)}")
    
    # If Gemini also fails, use direct LLM call with a specific prompt
    system_message = SystemMessage(content="""You are Thamizha, a friendly Tamil Nadu tourism guide.

You MUST provide helpful information about Tamil Nadu tourism, especially about the specific query.
Start with 'Vanakkam!' and be enthusiastic about Tamil Nadu.
Include specific places, attractions, or cultural aspects relevant to the query.
Don't mention that you're an AI. Just provide the information as a knowledgeable tour guide would.""")

    human_message = HumanMessage(content=query)
    
    try:
        response = llm.invoke([system_message, human_message])
        return response.content
    except Exception as e:
        print(f"LLM direct call error: {str(e)}")
        
        # Ultimate fallback if everything else fails - use a generic response
        return "Vanakkam! I'd be happy to help you explore the wonders of Tamil Nadu. Could you please try asking your question again in a different way?"

# ===============================================================
# TEXT-TO-SPEECH CONFIGURATION
# ===============================================================

# Audio settings
SAMPLE_RATE = 16000

# Thread pool for handling concurrent operations
executor = ThreadPoolExecutor(max_workers=2)

def generate_free_tts(text):
    """Generate audio using a free TTS API.

    Uses the StreamElements API to convert text to speech with the 'Brian' voice.

    Args:
        text (str): The text to convert to speech

    Returns:
        bytes or None: Audio data in bytes if successful, None otherwise
    """
    try:
        # Limit text length to 300 characters to avoid 400 errors
        if len(text) > 300:
            # Take first 300 chars and add ellipsis
            text = text[:297] + "..."
            print(f"Text truncated to {len(text)} characters for free TTS API")

        url = "https://api.streamelements.com/kappa/v2/speech"
        params = {
            "voice": "Brian",  # Using Brian voice for natural sounding speech
            "text": text
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.content
        else:
            print(f"Free TTS API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error with free TTS: {str(e)}")
        return None

# ===============================================================
# FLASK ROUTES
# ===============================================================

@app.route('/')
def index():
    """Render the main page of the application.

    Generates an introduction text and audio for the initial greeting.

    Returns:
        rendered template: The index.html template with introduction data
    """
    intro_data = generate_introduction()
    return render_template('index.html',
                         introduction=intro_data['text'],
                         intro_audio=intro_data['audio'])

def generate_introduction():
    """Generate an introduction text and audio for the initial greeting.

    Creates a personalized introduction for the Tamil Nadu tourism guide and
    attempts to generate audio for it using available TTS services.

    Returns:
        dict: Dictionary containing the introduction text and audio data (base64 encoded)
    """
    try:
        # Create introduction for Tamil Nadu tourism guide
        introduction = """Vanakkam! I'm Thamizha, your friendly Tamil Nadu tourism guide. I can help you discover the beautiful temples, beaches, hill stations, and cultural experiences that Tamil Nadu has to offer. Ask me about places to visit, transportation options, accommodation, local cuisine, or anything else you'd like to know about exploring Tamil Nadu!"""

        # Try ElevenLabs first (preferred for higher quality)
        if ELEVEN_LABS_API_KEY and ELEVEN_LABS_API_KEY != "your-elevenlabs-key":
            try:
                # Use cached_text_to_speech which uses ElevenLabs
                audio_data = cached_text_to_speech(introduction)
                if audio_data:
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    print("Successfully generated introduction audio with ElevenLabs")
                    return {
                        'text': introduction,
                        'audio': audio_b64
                    }
            except Exception as el_error:
                print(f"ElevenLabs error: {str(el_error)}")
        else:
            print("ElevenLabs API key not configured, falling back to free TTS")

        # Try free TTS as fallback
        try:
            audio_data = generate_free_tts(introduction)
            if audio_data:
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                print("Successfully generated introduction audio with free TTS")
                return {
                    'text': introduction,
                    'audio': audio_b64
                }
        except Exception as free_error:
            print(f"Free TTS error: {str(free_error)}")

        # If all TTS options fail, return text only
        return {
            'text': introduction,
            'audio': None
        }
    except Exception as e:
        print(f"Error generating introduction: {str(e)}")
        # Fallback introduction if everything fails
        return {
            'text': "Vanakkam! I'm Thamizha, your Tamil Nadu tourism guide. I'm here to help you explore the wonders of Tamil Nadu.",
            'audio': None
        }

def format_to_markdown(text):
    """Format the response text to Markdown for better display.
    
    Args:
        text (str): The raw response text
        
    Returns:
        str: Formatted Markdown text
    """
    # First, clean up any existing formatting issues
    text = re.sub(r'\d+\.\s*$', '', text)  # Remove dangling numbers
    text = re.sub(r'\s+\d+\.\s*', '\n\n', text)  # Fix numbered list formatting
    
    # Split text by section headers that might be in the text
    sections = re.split(r'((?:Famous Temples|Stunning Beaches|Rich Cultural Heritage|Beautiful Hill Stations|Delicious Local Cuisine))', text)
    
    formatted_sections = []
    current_section = ""
    
    for i, section in enumerate(sections):
        if i == 0:
            # First part is usually introduction
            formatted_sections.append(section.strip())
        elif section in ["Famous Temples", "Stunning Beaches", "Rich Cultural Heritage", 
                         "Beautiful Hill Stations", "Delicious Local Cuisine"]:
            # This is a section header
            if current_section:
                formatted_sections.append(current_section.strip())
            current_section = f"## {section}"
        else:
            # This is section content
            current_section += "\n\n" + section.strip()
    
    # Add the last section
    if current_section:
        formatted_sections.append(current_section.strip())
    
    # Join all formatted sections
    formatted_text = "\n\n".join(formatted_sections)
    
    # Fix numbered lists
    # First, identify potential numbered list items
    lines = formatted_text.split('\n')
    in_list = False
    list_indent = 0
    
    for i in range(len(lines)):
        # Check if line starts with a number followed by period
        if re.match(r'^\s*\d+\.\s', lines[i]):
            if not in_list:
                # Start of a new list
                in_list = True
                # Add a blank line before the list if needed
                if i > 0 and lines[i-1].strip():
                    lines[i-1] += '\n'
            
            # Ensure consistent formatting for list items
            number = re.match(r'^\s*(\d+)\.', lines[i]).group(1)
            content = re.sub(r'^\s*\d+\.\s*', '', lines[i])
            lines[i] = f"{number}. {content}"
        elif in_list and lines[i].strip():
            # Continuation of list item content
            lines[i] = "   " + lines[i].strip()
        elif in_list and not lines[i].strip():
            # Empty line ends the list
            in_list = False
    
    formatted_text = '\n'.join(lines)
    
    # Format attractions as bullet points
    place_names = [
        "Meenakshi Amman Temple", "Brihadeeswara Temple", "Ramanathaswamy Temple",
        "Ooty", "Kodaikanal", "Yelagiri", "Marina Beach", "Mahabalipuram", "Kanyakumari",
        "Chennai", "Madurai", "Thanjavur", "Rameshwaram", "Udhagamandalam", "Coaker's Walk",
        "Vivekananda Rock Memorial", "Pamban Bridge", "Thanjavur Palace", "Kumbakonam",
        "Kanyakumari Temple", "Ulagalandha Perumal Temple", "Desnoymeswarar Temple"
    ]
    
    # Convert place descriptions to bullet points
    lines = formatted_text.split('\n')
    for i in range(len(lines)):
        for place in place_names:
            if place in lines[i] and not re.match(r'^#|^\d+\.|^-\s', lines[i]):
                # This line contains a place name and is not already a heading, numbered list item, or bullet
                # Convert to bullet point
                lines[i] = "- " + lines[i]
                break
    
    formatted_text = '\n'.join(lines)
    
    # Format important information with bold
    # Format prices (₹XXX, ₹XXX-XXX)
    formatted_text = re.sub(r'(₹\d+(?:-\d+)?)', r'**\1**', formatted_text)
    
    # Format times (e.g., 5:00 AM, 10:00 PM)
    formatted_text = re.sub(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', r'**\1**', formatted_text)
    
    # Format dates and months
    formatted_text = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', r'**\1**', formatted_text)
    
    # Format important place names
    for place in place_names:
        formatted_text = re.sub(r'\b' + re.escape(place) + r'\b', r'**' + place + r'**', formatted_text)
    
    # Add horizontal rules between major sections
    formatted_text = re.sub(r'\n(## [^\n]+)\n', r'\n\n---\n\n\1\n', formatted_text)
    
    # Ensure proper spacing
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    # Fix any remaining formatting issues
    # Remove any trailing numbers at the end of paragraphs
    formatted_text = re.sub(r'\s+\d+\s*$', '', formatted_text)
    
    # Ensure proper spacing after headings
    formatted_text = re.sub(r'(##[^\n]+)\n([^\n])', r'\1\n\n\2', formatted_text)
    
    # Ensure proper spacing between list items
    formatted_text = re.sub(r'(\n- [^\n]+)\n([^\n-])', r'\1\n\n\2', formatted_text)
    
    return formatted_text

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat messages from the user."""
    user_message = request.json.get('message')
    summarize = request.json.get('summarize', False)  # New parameter to control summarization

    try:
        # Limit incoming message length
        if len(user_message) > 500:
            user_message = user_message[:500] + "..."
            
        print(f"Processing user message: {user_message}")

        # Process message through LangChain agent
        try:
            print("Attempting to use agent_executor")
            response = agent_executor.invoke({
                "input": user_message,
                "chat_history": []
            })
            response_text = response.get("output", "")
            
            # Check if response is empty or too short
            if not response_text or len(response_text) < 20:
                print("Agent response too short, falling back")
                response_text = fallback_response(user_message)
                
            response_text = clean_response(response_text)
            print(f"Agent response length: {len(response_text)}")
        except Exception as agent_error:
            print(f"Agent error: {str(agent_error)}")
            response_text = fallback_response(user_message)
            response_text = clean_response(response_text)
            print(f"Fallback response length: {len(response_text)}")

        # Process with Gemini if summarization is requested
        if summarize and len(response_text) > 600:  # Only summarize longer responses
            try:
                processed_response = process_with_gemini(response_text)
                summary = processed_response['summary']
                full_response = processed_response['full_response']
                
                # Format responses as Markdown
                summary = format_to_markdown(summary)
                full_response = format_to_markdown(full_response)
                
                # Use summary for display and audio
                display_text = summary
                audio_text = summary.replace('**', '').replace('#', '')
                
                # Include full response in the return data
                has_more = len(full_response) > len(summary)
            except Exception as gemini_error:
                print(f"Gemini summarization error: {str(gemini_error)}")
                # If summarization fails, use the original response
                display_text = format_to_markdown(response_text)
                audio_text = display_text.replace('**', '').replace('#', '')
                has_more = False
                full_response = None
        else:
            # Format the response as Markdown
            display_text = format_to_markdown(response_text)
            audio_text = display_text.replace('**', '').replace('#', '')
            has_more = False
            full_response = None

        # For longer responses, use only the first part for audio
        if len(audio_text) > 500:
            audio_text = audio_text[:500] + "..."
            print(f"Audio text truncated to {len(audio_text)} characters for TTS")

        # Generate audio
        audio_data = None
        if ELEVEN_LABS_API_KEY and ELEVEN_LABS_API_KEY != "your-elevenlabs-key":
            try:
                audio_data = cached_text_to_speech(audio_text)
                if audio_data:
                    print("Successfully generated audio with ElevenLabs")
            except Exception as el_error:
                print(f"ElevenLabs error in chat: {str(el_error)}")
        else:
            print("ElevenLabs API key not configured, falling back to free TTS")

        if not audio_data:
            try:
                audio_data = generate_free_tts(audio_text)
                if audio_data:
                    print("Successfully generated audio with free TTS")
            except Exception as free_error:
                print(f"Free TTS error in chat: {str(free_error)}")

        # Return response with audio if available
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'response': display_text,
                'audio': audio_b64,
                'has_more': has_more,
                'full_response': full_response,
                'status': 'success'
            })
        else:
            return jsonify({
                'response': display_text,
                'audio': None,
                'has_more': has_more,
                'full_response': full_response,
                'status': 'no_audio'
            })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        # Use Gemini for ultimate fallback
        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
                model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"""You are Thamizha, a friendly Tamil Nadu tourism guide.
                
Generate a helpful response to the following query about Tamil Nadu tourism:
"{user_message}"

Start with 'Vanakkam!' and be enthusiastic about Tamil Nadu.
Your response should be informative, conversational, and focused on Tamil Nadu tourism.
Include specific places, attractions, or cultural aspects relevant to the query.
"""
                
                response = model.generate_content(prompt)
                if response and response.text:
                    return jsonify({
                        'response': format_to_markdown(response.text),
                        'audio': None,
                        'status': 'recovered_error'
                    })
        except Exception as gemini_error:
            print(f"Gemini fallback error: {str(gemini_error)}")
        
        # If all else fails, return a simple message
        return jsonify({
            'response': "Vanakkam! I'd be happy to help you explore the wonders of Tamil Nadu. Could you please try asking your question again in a different way?",
            'audio': None,
            'status': 'recovered_error'
        })

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_with_assemblyai():
    """Transcribe audio using AssemblyAI's speech recognition API.

    Receives an audio file, saves it temporarily, transcribes it using
    AssemblyAI, and returns the transcription text.

    Returns:
        JSON: Transcription result or error message
    """
    # Check if audio file was provided
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded audio file temporarily
    audio_file = request.files['audio']
    temp_file_path = "temp_recording.wav"
    audio_file.save(temp_file_path)

    try:
        # Verify AssemblyAI API key is configured
        if not ASSEMBLY_AI_API_KEY or ASSEMBLY_AI_API_KEY == "your-assemblyai-key":
            print("No valid AssemblyAI API key found")

            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            return jsonify({
                'transcript': "",
                'error': "AssemblyAI API key is not configured. Please update api_key.json with your key.",
                'status': 'error'
            }), 400

        # Use AssemblyAI for transcription
        print("Using AssemblyAI for transcription")

        # Ensure AssemblyAI is configured with the API key
        aai.settings.api_key = ASSEMBLY_AI_API_KEY

        # Create transcriber and process the audio file
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        text = transcript.text or ""

        print(f"AssemblyAI transcription result: {text}")

        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Return the successful transcription
        return jsonify({
            'transcript': text,
            'status': 'success'
        })
    except Exception as e:
        print(f"Error in AssemblyAI transcription: {str(e)}")

        # Clean up the temporary file even if an error occurred
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Return error information
        return jsonify({
            'transcript': "",
            'error': f"AssemblyAI error: {str(e)}",
            'status': 'error'
        }), 500

@lru_cache(maxsize=128)
def cached_text_to_speech(text):
    """Convert text to speech using ElevenLabs with caching.

    Uses LRU cache to avoid regenerating audio for the same text multiple times.

    Args:
        text (str): The text to convert to speech

    Returns:
        bytes: Audio data in bytes or None if generation fails
    """
    if not ELEVEN_LABS_API_KEY or ELEVEN_LABS_API_KEY == "your-elevenlabs-key":
        print("ElevenLabs API key not properly configured")
        return None

    try:
        # Limit text length to avoid excessive API usage
        if len(text) > 1000:
            text = text[:1000] + "..."
            print(f"Text truncated to {len(text)} characters for ElevenLabs API")

        # Generate audio using ElevenLabs
        audio_data = client.generate(
            text=text,
            voice="xnx6sPTtvU635ocDt2j7",  # Specific voice ID from ElevenLabs
            model="eleven_multilingual_v2",  # Multilingual model for better pronunciation
            voice_settings=VoiceSettings(stability=0.75, similarity_boost=0.75)  # Voice clarity settings
        )

        # Convert audio data to bytes if it's a generator
        if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
            audio_data = b''.join(audio_data)

        return audio_data
    except Exception as e:
        print(f"Error in ElevenLabs TTS generation: {str(e)}")
        return None

# Add this function to process responses through Gemini
def process_with_gemini(text, max_words=500):
    """Process text through Gemini API and summarize to specified word count.
    
    Args:
        text (str): The text to process
        max_words (int): Maximum number of words for the summary
        
    Returns:
        dict: Dictionary with 'summary' and 'full_response'
    """
    try:
        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-pro')
            
            # Create prompt for summarization with Markdown formatting
            prompt = f"""Summarize the following text in approximately {max_words} words while preserving the most important information about Tamil Nadu tourism.

MAINTAIN THE ORIGINAL MARKDOWN FORMATTING:
• Keep all section headings (## and ###)
• Preserve **bold** text for important information
• Keep bullet points and numbered lists
• Maintain horizontal rules (---)
• Ensure proper spacing between sections

Here's the text to summarize:

{text}"""
            
            # Generate summary
            response = model.generate_content(prompt)
            summary = response.text
            
            return {
                'summary': summary,
                'full_response': text,
                'status': 'success'
            }
        else:
            print("Gemini API not available, returning original text")
            return {
                'summary': text[:max_words*6],  # Approximate word count based on average word length
                'full_response': text,
                'status': 'no_gemini'
            }
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        return {
            'summary': text[:max_words*6],  # Fallback to simple truncation
            'full_response': text,
            'status': 'error'
        }

@app.route('/get_full_response', methods=['POST'])
def get_full_response():
    """Return the full response when user requests more details."""
    try:
        full_response = request.json.get('full_response')
        
        if not full_response:
            return jsonify({
                'response': "I'm sorry, the full response is not available.",
                'status': 'error'
            })
            
        # Generate audio for the full response
        audio_text = full_response.replace('**', '').replace('#', '')
        
        # Limit audio text length
        if len(audio_text) > 500:
            audio_text = audio_text[:500] + "... (continued in text)"
            
        # Generate audio
        audio_data = None
        if ELEVEN_LABS_API_KEY and ELEVEN_LABS_API_KEY != "your-elevenlabs-key":
            try:
                audio_data = cached_text_to_speech(audio_text)
            except Exception:
                pass
                
        if not audio_data:
            try:
                audio_data = generate_free_tts(audio_text)
            except Exception:
                pass
                
        # Return full response with audio if available
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'response': full_response,
                'audio': audio_b64,
                'status': 'success'
            })
        else:
            return jsonify({
                'response': full_response,
                'audio': None,
                'status': 'no_audio'
            })
            
    except Exception as e:
        print(f"Error in get_full_response: {str(e)}")
        return jsonify({
            'response': "I'm sorry, I encountered an error retrieving the full response.",
            'status': 'error'
        })

# ===============================================================
# APPLICATION ENTRY POINT
# ===============================================================

if __name__ == '__main__':
    # Run the Flask application
    print("Starting Tamilnadu Tourism Guide...")
    print("Access the application at http://localhost:5000")
    app.run()
