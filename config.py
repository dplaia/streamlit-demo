from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from typing import Final, Optional

class Config(BaseSettings):
    """Global configuration settings"""
    MAX_RETRIES: int = Field(default=3)
    REQUEST_TIMEOUT: int = Field(default=10)
    CRAWL_CONCURRENCY: int = Field(default=5)
    PAPERS_PER_PAGE: int = Field(default=200)
    
    # API endpoints
    PERPLEXITY_BASE_URL: str = Field(default="https://api.perplexity.ai")
    SERPER_BASE_URL: str = Field(default="https://google.serper.dev/search")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")

    SERPER_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'SERPER_API_KEY'})
    PERPLEXITY_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'PERPLEXITY_API_KEY'})
    FIRECRAWL_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'FIRECRAWL_API_KEY'})
    GEMINI_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'GEMINI_API_KEY'})
    DEEPSEEK_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'DEEPSEEK_API_KEY'})
    OPENROUTER_API_KEY: Optional[str] = Field(None, json_schema_extra={'env ': 'OPENROUTER_API_KEY'})

    PAPERS_WITH_CODE_CSRF_TOKEN: str = Field(default="2ix1PR0FtUWIW5ePo08I3vhgHsvJ6fpqj0x1Ijjo4egxiofnUBzkX67bnHwbNd8G")

    # Tool Descriptions
    TOOL_DESCRIPTIONS: Final = {
        "google_general": "General web search using Google via Serper API",
        "scholar": "Academic search using Google Scholar",
        "perplexity": "LLM-powered search with Perplexity AI",
        "papers_with_code": "Research paper search with code implementations"
    }
    
    # Model Settings
    FLASH1_MODEL: str = Field(default="gemini-1.5-flash") # (free)
    FLASH2_MODEL: str = Field(default="gemini-2.0-flash") # (free, 1500 req/day, 15 RPM)
    PRO2_MODEL: str = Field(default="gemini-2.0-pro-exp-02-05") # (free, 50 req/day, 2 RPM)
    LITE2_MODEL: str = Field(default="gemini-2.0-flash-lite-preview-02-05") # (free, 1500 req/day, 30 RPM)
    FLASH2T_MODEL: str = Field(default='gemini-2.0-flash-thinking-exp-01-21') # (free)
    DEEPSEEK_R1: str = Field(default="deepseek-reasoner") # (paid)
    OPENROUTER_DEEPSEEK_R1_FREE: str = Field(default="deepseek/deepseek-r1:free") # OpenRouter (free)
    OPENROUTER_DEEPSEEK_R1: str = Field(default="deepseek/deepseek-r1") # OpenRouter (paid)
    OPENROUTER_PERPLEXITY_SONAR_REASONING: str = Field(default="perplexity/sonar-reasoning") # OpenRouter (paid)

    BASEAGENT_MODEL:str = Field(default='gemini-2.0-flash') 

    model_config = ConfigDict(env_file = ".env", extra = "ignore")
    # class Config:
    #     env_file = ".env"
    #     extra = "ignore"
