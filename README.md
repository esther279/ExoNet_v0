# ExoNet: Idea Illustration & Feasiblity 

## Overview

This simple multi-agent framework in `chat_multi.py` offers collaborative AI agents with their own specialized backgrounds/expertise and flexible interaction order!


## Key Features

### 🎯 **Individual Agent Backgrounds**
- **Specialty Definitions**: Clear role and expertise areas for each agent
- **Model Configuration**: Individual model settings per agent
- **PDFs**: Each agent can have access to specific PDF documents
- **Images**: Visual context through image files
- **Websites**: Web content as background knowledge
- **Text Files**: Additional text-based context


### 🔄 **Flexible Interaction Patterns**
- **Custom Interaction Order**: Define how agents collaborate, e.g., [1, 3, 2, 4]
- **Multiple Rounds**: Configurable deliberation cycles. Each round contains the entire list of interaction order. For example, two rounds would be [1, 3, 2, 4] and [1, 3, 2, 4]
- **Collaborative Learning**: Agents build upon each other's responses
- **Human interaction**: After running default_question for the specified number of rounds, user can prompt new questions. 



## Usage

### Command Line Interface
```bash
# Basic usage with config file
python chat_multi.py --config config_example.json

# Enable verbose output
python chat_multi.py --config config_example.json --verbose

```


### Example Configuration
```json
{
  "title": "Enhanced Multi-Agent System Demo",
  "description": "Multi-agent system with individual backgrounds, specialties, and flexible interaction order",
  "max_tokens": 128000,
  "rounds": 5,
  "clean_slate": true,
  "interaction_order": [1],
  "default_question": "Which is more important in DOE national laboratories, choose one: (A) data and AI security, (B) develop and deploy AI agents for science",
  "agents": [
    {
      "name": "Scientist",
      "model": "gpt-5",
      "specialty": "You are a senior chemist with also expertise in computer science, You are a chemist with also expertise in computer science, allowing you to be visionary and bridge the gap between physical science and computational science/AI. Provide a clear, concise proposal to the user's question. When provided, incorporate the attached document/image.",
      "char_limit": 3000,
      "websites": [ 
        {
            "url": "https://www.bnl.gov/staff/kyager",
            "name": "Staff",
            "crawl_depth": 1,
            "max_crawl_pages": 1
        },   
        {
            "url": "http://yager-research.ca/",
            "name": "KY-research",
            "crawl_depth": 1,
            "max_crawl_pages": 10
        }
        ],
      "pdfs": ["data/KY/Yager_Exocortex.pdf"]
    },
    {
      "name": "Computational Scientist",
      "model": "gpt-5",
      "system_prompt": "You are a computational scientist with a foundation in engineering and economics, committed to efficient, cost-effective approaches that accelerate scientific progress while fostering freedom and creativity for researchers.",
      "websites": ["https://www.bnl.gov/staff/etsai"],
      "images": [ "data/ET/vision_overview.png"  ]
    },
  ],
}

```


### Suggestions
- **Azure OpenAI**: The implementation employs direct calls via Azure OpenAI: API key, base, deployment name should be specified accordingly. See line 370 in `chat_multi.py` and specialize in `~/.bashrc` e.g. export AZURE_API_KEY=XXX
- **max_tokens**: Higher max_tokens (6000-8000) for detailed analysis, while lower max_tokens for concise responses. However note that if too constrained, model may have trouble outputting reasonable response. 
- **char_limit** For background information, web crawling, char_limit (e.g. 2000) should be chosen depending on the number of documents and the context window.
- **clean_slate**: Starting fresh in every round allows for convenient repetition of the question across multiple rounds, acquiring statistical information to account for the model stochasticity.


