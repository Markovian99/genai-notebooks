QUESTIONS_10K = ["Describe {company_name}'s business.", "Describe what {company_name} does.", "What sector or industry does {company_name} operate in?",
                        "What market does {company_name} serve?", "What products does {company_name} offer?", "What services does the {company_name} offer?", "Who are {company_name}'s clients or customers?", "Who are the suppliers for {company_name}?",
                        "What is the revenue for {company_name}?", "What is the net income for {company_name}?","What is the operating income for {company_name}?","What's {company_name}'s EBITDA?",
                        "What is the gross profit for {company_name}?", "What's {company_name}'s gross margin like?", "How much cash does {company_name} have?", 
                        "What are the key risk factors {company_name} is facing?", "What did management discuss in the 10K for {company_name}","How much revenue did {company_name} earn in the last quarter?"]

QUESTIONS_STATIC_STOCK = ["What is the current stock price for {company_name}?", "What is the stock price for {company_name}?", "What is the stock price for {company_name} today?",
                          "What is the latest stock price for {company_name}?", "What was the trading volume for {company_name}?", "What was the open price for {company_name}?",
                          "What was the close price for {company_name}?", "What was the high price for {company_name}?", "What was the low price for {company_name}?"]


QUESTIONS_AGENT_STOCK = ["What was the average stock price for {company_name} in the last 30 days?", "What was the average stock price for {company_name} in the last 90 days?",
                         "What was the average trading volume for {company_name} in the last 30 days?", "What was the average high low price difference for {company_name} in the last 30 days?",
                         "What was the volume weighted average price for {company_name} in the last 30 days?", "What was the peak stock price for {company_name} in the last 30 days?",
                         "What was the average open to close price difference for {company_name} in the last 30 days?", "What was the standard deviation of trading volume for {company_name} over the last 30 days?",]

QUESTIONS_NEWS = ["What are the recent news headlines for {company_name}?", "Has any news been published about {company_name} recently?", "What are the latest news articles about {company_name}?",
                  "What recent posts on twitter are about {company_name}?"]

QUESTIONS_GENERIC = ["For {company_name}, tell me more about that.", "For {company_name}, what else can you tell me about that?","For {company_name}, what else do you know?"]


ROUTE_DICT = {"10K": QUESTIONS_10K, "Stock": QUESTIONS_STATIC_STOCK, "AgentStock": QUESTIONS_AGENT_STOCK, "News": QUESTIONS_NEWS, "Generic": QUESTIONS_GENERIC}


ROUTING_PROMPT = """
You are a helpful assistant determining how to route a question or prompt. Please give the category of the question or prompt by learning from the examples below.
"""

for idx, key in enumerate(ROUTE_DICT.keys()):
    ROUTING_PROMPT += f"\nCategory #{idx+1}:\n"
    for question in ROUTE_DICT[key]:
        ROUTING_PROMPT += question + "\n"
ROUTING_PROMPT += "\nPlease give the category of the question or prompt by learning from the examples above.\n"
