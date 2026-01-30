"""
Shopping Agent - Agentic Personal Stylist Recommender
Uses LangChain + Groq (Llama 3.3 70B) + Direct Store Links

Design Patterns:
1. Tool Use: TavilySearchResults for product discovery
2. Reflection: LLM generates product recommendations based on search context
3. Direct Links: Generate store-specific search URLs for reliable shopping
"""

import os
import json
import urllib.parse
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults


# ==========================================
# STEP 1: Pydantic Data Models
# ==========================================

class Product(BaseModel):
    """Schema for a single product recommendation."""
    title: str = Field(description="Product title/name")
    price: str = Field(description="Estimated price range")
    link: str = Field(description="Direct URL to the product or search")
    image_url: str = Field(description="Product image URL or placeholder")
    store_name: str = Field(description="Store name (e.g., Myntra, Amazon)")
    color_match: str = Field(description="How well the color matches (Exact/Close/Similar)")


class StoreLink(BaseModel):
    """Schema for direct store search links."""
    store_name: str
    search_url: str
    logo_url: str


class ProductList(BaseModel):
    """Schema for the list of product recommendations."""
    products: List[Product] = Field(description="List of product recommendations")
    query_used: str = Field(default="", description="The search query used")
    total_found: int = Field(default=0, description="Total products found")
    filtered_count: int = Field(default=0, description="Number of products after filtering")


# ==========================================
# STEP 2: Direct Store Link Generators
# ==========================================

def _generate_store_links(color: str, item_type: str) -> List[dict]:
    """
    Generate direct search URLs for major Indian e-commerce stores.
    These links go directly to search results with the exact query.
    """
    # Normalize item type - be specific to avoid t-shirt/shirt confusion
    item_mapping = {
        "shirt": "formal shirts",
        "t-shirt": "t-shirts",
        "kurta": "kurta",
        "jacket": "jacket",
        "pants": "trousers",
        "jeans": "jeans"
    }
    
    # Get specific item term or use as-is
    specific_item = item_mapping.get(item_type.lower(), item_type)
    
    # Build search queries
    query = f"{color} {specific_item} men"
    encoded_query = urllib.parse.quote(query)
    
    # Myntra uses specific search URL format
    myntra_query = urllib.parse.quote(f"{color} {specific_item}")
    
    stores = [
        {
            "store_name": "Myntra",
            "search_url": f"https://www.myntra.com/search?rawQuery={myntra_query}&gender=men",
            "logo_url": "https://logos-world.net/wp-content/uploads/2022/12/Myntra-Logo.png",
            "description": f"{color} {specific_item.title()}"
        },
        {
            "store_name": "Amazon",
            "search_url": f"https://www.amazon.in/s?k={encoded_query}",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
            "description": f"{color} {specific_item.title()}"
        },
        {
            "store_name": "Ajio",
            "search_url": f"https://www.ajio.com/search/?text={encoded_query}",
            "logo_url": "https://logos-download.com/wp-content/uploads/2022/01/AJIO_Logo.png",
            "description": f"{color} {specific_item.title()}"
        },
        {
            "store_name": "Flipkart",
            "search_url": f"https://www.flipkart.com/search?q={encoded_query}",
            "logo_url": "https://static-assets-web.flixcart.com/fk-p-linchpin-web/fk-cp-zion/img/flipkart-plus_8d85f4.png",
            "description": f"{color} {specific_item.title()}"
        }
    ]
    
    return stores


# ==========================================
# STEP 3: Environment & LLM Setup
# ==========================================

def _check_api_keys():
    """Verify required API keys are set."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not tavily_key:
        raise EnvironmentError("TAVILY_API_KEY not found")
    
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not found")
    
    return True


def _get_llm():
    """Initialize Groq LLM with Llama 3.3 70B Versatile."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2048
    )


def _get_search_tool():
    """Initialize Tavily Search for product research."""
    return TavilySearchResults(
        max_results=8,
        search_depth="advanced",
        include_answer=False
    )


# ==========================================
# STEP 4: Product Recommendation Chain
# ==========================================

def _create_recommendation_chain(llm):
    """
    Create a chain that generates product recommendations based on
    search results and seasonal context.
    """
    
    parser = JsonOutputParser(pydantic_object=ProductList)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Personal Stylist AI. Generate product recommendations based on the user's request.

CRITICAL RULES:
1. EXACT ITEM TYPE: If user requests "Shirt", recommend ONLY formal/casual SHIRTS (NOT t-shirts!)
   - Shirt = Formal shirts, casual shirts, dress shirts
   - T-Shirt = Round neck, polo, graphic tees
2. Generate realistic product TITLES for Indian stores (Myntra, Amazon, Ajio)
3. Provide estimated PRICE in INR based on typical Indian prices
4. Use the provided store search URLs as links - DO NOT invent URLs
5. Use image: "https://via.placeholder.com/300x400?text=View+on+Store"

Return valid JSON. No markdown."""),
        
        ("human", """Season: {season}
Color: {color}
Item Type: {item_type} (ONLY recommend THIS exact item type, NOT different items!)
Budget: {budget}

Available store links:
{store_links}

Generate exactly 4 {item_type} recommendations (NOT other item types) that match the {season} aesthetic.
Each product MUST be a {item_type}, not anything else.

{format_instructions}""")
    ])
    
    return prompt | llm | parser, parser


# ==========================================
# STEP 5: Main Agent Function
# ==========================================

def get_agent_recommendations(
    season: str,
    color: str,
    item_type: str,
    budget: str = "medium"
) -> dict:
    """
    Main agent function - uses multi-step approach:
    1. Search for product context
    2. Generate recommendations with direct store links
    """
    
    try:
        _check_api_keys()
    except EnvironmentError as e:
        return {"error": str(e), "products": [], "store_links": []}
    
    # Step 1: Generate direct store links (guaranteed to work)
    store_links = _generate_store_links(color, item_type)
    
    print(f"[AGENT] Generating recommendations for {color} {item_type}")
    
    # Step 2: Generate recommendations using LLM
    llm = _get_llm()
    chain, parser = _create_recommendation_chain(llm)
    
    try:
        result = chain.invoke({
            "season": season,
            "color": color,
            "item_type": item_type,
            "budget": budget,
            "store_links": json.dumps(store_links, indent=2),
            "format_instructions": parser.get_format_instructions()
        })
        
        # Ensure products use valid store links
        for product in result.get("products", []):
            # Find matching store link
            store = product.get("store_name", "").lower()
            for sl in store_links:
                if sl["store_name"].lower() in store.lower():
                    product["link"] = sl["search_url"]
                    break
            else:
                # Default to first store if no match
                product["link"] = store_links[0]["search_url"]
        
        result["store_links"] = store_links
        result["query_used"] = f"{color} {item_type}"
        result["filtered_count"] = len(result.get("products", []))
        
        print(f"[AGENT] Generated {result['filtered_count']} recommendations")
        return result
        
    except Exception as e:
        print(f"[AGENT] Recommendation failed: {str(e)}")
        # Fallback: return just the store links
        return {
            "error": None,
            "products": [],
            "store_links": store_links,
            "query_used": f"{color} {item_type}",
            "message": "Browse stores directly for best results"
        }


# ==========================================
# Convenience Functions
# ==========================================

def get_quick_shop_links(color: str, item_type: str) -> List[dict]:
    """Get direct shopping links without LLM processing."""
    return _generate_store_links(color, item_type)


if __name__ == "__main__":
    print("Testing Shopping Agent...")
    result = get_agent_recommendations(
        season="Deep Autumn",
        color="Olive Green",
        item_type="shirt",
        budget="medium"
    )
    print(json.dumps(result, indent=2))
