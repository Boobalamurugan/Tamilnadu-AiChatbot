# Tamil Nadu Tourism Guide Chatbot

A conversational AI chatbot that provides information about Tamil Nadu tourism, attractions, culture, and travel tips.

## Features

- **Natural Language Understanding**: Uses Groq LLM with Llama 3.1 model for natural language processing
- **Enhanced Search Capabilities**: Integrates TavilySearchResults for detailed and up-to-date information about Tamil Nadu
- **Backup Search Tools**: Includes Wikipedia and DuckDuckGo as fallback search options
- **Voice Interaction**: Supports both text and voice interactions
  - Text-to-Speech: Uses ElevenLabs for high-quality voice output
  - Speech Recognition: Uses AssemblyAI for accurate speech-to-text conversion
- **Modern UI/UX**:
  - Responsive design with Tailwind CSS
  - Smooth animations and transitions
  - Visual feedback for all interactions
  - Mobile-friendly interface
- **Robust Architecture**:
  - Fallback mechanisms for API service disruptions
  - Graceful error handling for questions outside its knowledge domain
  - Configurable API settings
  - Scalable Flask backend

## Prerequisites

- Python 3.8+
- Flask web framework
- Internet connection for API access
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Git (for cloning repository)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TamilChatbot
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure API keys:
   Create a `.env` file in the project root with the following:
   ```
   GROQ_API_KEY=your-groq-api-key
   ELEVEN_LABS_API_KEY=your-elevenlabs-key
   ASSEMBLY_AI_API_KEY=your-assemblyai-key
   TAVILY_API_KEY=your-tavily-api-key
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

5. Access the interface:
   ```
   http://127.0.0.1:5000/
   ```

## Required API Keys

Register for free/paid API keys from:
- [Groq API](https://console.groq.com/) - For LLM access with Llama model
- [Tavily API](https://tavily.com/) - For enhanced search capabilities
- [AssemblyAI](https://www.assemblyai.com/dashboard/signup) - For speech recognition
- [ElevenLabs](https://elevenlabs.io/sign-up) - For high-quality text-to-speech

## Voice Features

- **Intelligent Text-to-Speech**:
  - Primary: StreamElements API for fast, reliable synthesis
  - Backup: ElevenLabs for premium quality when available
- **Advanced Speech Recognition**:
  - High-accuracy transcription with AssemblyAI
  - Real-time voice activity detection
  - Automatic punctuation and formatting
- **Intuitive Controls**:
  - One-click recording toggle
  - Visual feedback during recording
  - Automatic silence detection

## Technology Stack

### Backend
- Flask (Python 3.8+)
- Groq LLM with Llama 3.1 model
- LangChain for agent framework
- Tavily Search API for enhanced search capabilities
- Wikipedia and DuckDuckGo search tools
- AssemblyAI Speech Recognition
- StreamElements & ElevenLabs TTS

### Frontend
- HTML5 & CSS3
- Tailwind CSS
- Modern JavaScript
- Font Awesome Icons
- Animate.css
- Particles.js for background effects

## Project Structure

```
TamilChatbot/
├── app.py                 # Main Flask application with LLM and search tools
├── templates/
│   └── index.html        # Frontend interface with responsive design
├── static/
│   ├── css/
│   │   └── style.css     # Additional CSS styles
│   └── js/
│       └── script.js     # Frontend logic for chat and voice features
├── .env                  # API credentials
└── requirements.txt      # Python dependencies
```

## Example Questions

The chatbot can answer questions about:

- "What are the must-visit temples in Tamil Nadu?"
- "Tell me about the best time to visit Ooty"
- "What are some famous dishes to try in Tamil Nadu?"
- "How can I travel from Chennai to Madurai?"
- "What are the entry fees for Mahabalipuram monuments?"
- "Tell me about Pongal festival"
- "What are some budget accommodation options in Kodaikanal?"

## Limitations

The chatbot is specifically designed to provide information about Tamil Nadu tourism. It may not be able to answer questions about:
- Current political figures or events
- Programming or technical topics
- Topics unrelated to Tamil Nadu tourism

## Troubleshooting Guide

### Microphone Issues
- Enable microphone permissions in browser settings
- Refresh page after granting permissions
- Try using Chrome or Firefox for best compatibility

### Audio Problems
- Verify API keys are correctly configured in `.env`
- Check browser audio output settings
- Ensure stable internet connection

### API Limitations
- Monitor API usage quotas
- Implement request rate limiting
- Check server logs for error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - feel free to use and modify for your own projects.

## Acknowledgements

This project uses several open-source libraries and APIs:

- [LangChain](https://www.langchain.com/) - For building the conversational agent
- [Groq](https://groq.com/) - For fast LLM inference with Llama model
- [Tavily](https://tavily.com/) - For enhanced search capabilities
- [Flask](https://flask.palletsprojects.com/) - For the web framework
- [Tailwind CSS](https://tailwindcss.com/) - For responsive design
- [AssemblyAI](https://www.assemblyai.com/) - For speech recognition
- [ElevenLabs](https://elevenlabs.io/) - For text-to-speech
