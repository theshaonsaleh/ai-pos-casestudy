"""
LLM Query Engine Module
Converts natural language questions to pandas queries with OpenAI API or mock fallback.
"""

import os
import pandas as pd
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()


class LLMQueryEngine:
    """Natural language query engine for POS data with offline mock support."""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Initialize the query engine.
        
        Args:
            api_key: OpenAI API key (if None, will try to load from env)
            use_mock: Force use of mock responses even if API key is available
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_mock = use_mock or not self.api_key
        
        if not self.use_mock:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model = "gpt-3.5-turbo"
            except ImportError:
                print("Warning: openai package not installed. Using mock mode.")
                self.use_mock = True
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}. Using mock mode.")
                self.use_mock = True
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a data analyst assistant for a point-of-sale system. 
Convert natural language questions about sales data into executable pandas code.

The DataFrame is called 'df' and has these columns:
- transaction_id: Unique transaction identifier
- timestamp: Transaction datetime
- date: Transaction date
- hour: Hour of day (0-23)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- day_name: Day name (Monday, Tuesday, etc.)
- product_name: Name of product
- category: Product category (Hot Beverage, Cold Beverage, Pastry, Food, Snack)
- quantity: Quantity sold (negative for refunds)
- unit_price: Price per unit
- total: Total amount (negative for refunds)
- cost: Cost of goods
- staff_id: Staff member ID
- staff_name: Staff member name
- payment_method: card or cash
- is_refund: Boolean, True if refund

IMPORTANT RULES:
1. Return ONLY executable Python code using pandas, no explanations
2. Use 'df' as the DataFrame name
3. Code must return a result (not just print)
4. Keep it simple and safe (no file operations, no imports)
5. For "top N" queries, use .nlargest() or .head()
6. For date filtering, use df[df['date'] == 'YYYY-MM-DD']
7. For aggregations, use .groupby().agg()

Examples:

Q: What were the top 5 products by revenue?
A: df.groupby('product_name')['total'].sum().nlargest(5)

Q: Show me sales by hour of day
A: df.groupby('hour')['total'].sum().sort_index()

Q: Which staff member had the most transactions?
A: df.groupby('staff_name')['transaction_id'].nunique().nlargest(1)

Q: What was the average transaction value on weekends?
A: df[df['day_of_week'].isin([5, 6])]['total'].mean()

Q: How many refunds did we have?
A: df[df['is_refund'] == True]['transaction_id'].nunique()

Now convert the user's question to pandas code."""
    
    def _generate_code_with_llm(self, question: str) -> str:
        """
        Generate pandas code using OpenAI API.
        
        Args:
            question: Natural language question
            
        Returns:
            Generated pandas code
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            code = response.choices[0].message.content.strip()
            
            # Remove markdown code fences if present
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def _generate_code_with_mock(self, question: str) -> str:
        """
        Generate pandas code using pattern matching (offline mock).
        
        Args:
            question: Natural language question
            
        Returns:
            Generated pandas code
        """
        question_lower = question.lower()
        
        # Pattern matching for common queries
        if 'top' in question_lower and 'product' in question_lower and 'revenue' in question_lower:
            n = 5
            for word in question.split():
                if word.isdigit():
                    n = int(word)
            return f"df.groupby('product_name')['total'].sum().nlargest({n})"
        
        elif 'top' in question_lower and 'product' in question_lower:
            n = 5
            for word in question.split():
                if word.isdigit():
                    n = int(word)
            return f"df.groupby('product_name')['total'].sum().nlargest({n})"
        
        elif 'sales by hour' in question_lower or 'hourly sales' in question_lower:
            return "df.groupby('hour')['total'].sum().sort_index()"
        
        elif 'sales by day' in question_lower and 'week' in question_lower:
            return "df.groupby('day_name')['total'].sum()"
        
        elif 'staff' in question_lower and ('most' in question_lower or 'top' in question_lower):
            return "df.groupby('staff_name')['transaction_id'].nunique().nlargest(1)"
        
        elif 'average' in question_lower and 'transaction' in question_lower:
            if 'weekend' in question_lower:
                return "df[df['day_of_week'].isin([5, 6])].groupby('transaction_id')['total'].sum().mean()"
            else:
                return "df.groupby('transaction_id')['total'].sum().mean()"
        
        elif 'refund' in question_lower and 'how many' in question_lower:
            return "df[df['is_refund'] == True]['transaction_id'].nunique()"
        
        elif 'total sales' in question_lower or 'total revenue' in question_lower:
            return "df[df['is_refund'] == False]['total'].sum()"
        
        elif 'best selling' in question_lower or 'most popular' in question_lower:
            return "df.groupby('product_name')['quantity'].sum().nlargest(5)"
        
        elif 'category' in question_lower and 'sales' in question_lower:
            return "df.groupby('category')['total'].sum().sort_values(ascending=False)"
        
        else:
            # Default: show summary statistics
            return "df[['total', 'quantity']].describe()"
    
    def query(self, df: pd.DataFrame, question: str) -> Dict:
        """
        Execute a natural language query on the dataframe.
        
        Args:
            df: DataFrame to query
            question: Natural language question
            
        Returns:
            Dictionary with result, code, and metadata
        """
        try:
            # Generate code
            if self.use_mock:
                code = self._generate_code_with_mock(question)
                method = "mock"
            else:
                code = self._generate_code_with_llm(question)
                method = "llm"
            
            # Execute code in a sandboxed environment
            local_vars = {'df': df.copy(), 'pd': pd}
            
            try:
                result = eval(code, {"__builtins__": {}}, local_vars)
            except Exception as exec_error:
                return {
                    'success': False,
                    'error': f"Execution error: {exec_error}",
                    'code': code,
                    'method': method
                }
            
            # Format result
            if isinstance(result, pd.Series):
                result_str = result.to_string()
                result_data = result.to_dict()
            elif isinstance(result, pd.DataFrame):
                result_str = result.to_string()
                result_data = result.to_dict()
            else:
                result_str = str(result)
                result_data = result
            
            return {
                'success': True,
                'result': result_data,
                'result_str': result_str,
                'code': code,
                'method': method,
                'question': question
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'method': 'mock' if self.use_mock else 'llm'
            }
    
    def ask(self, df: pd.DataFrame, question: str) -> str:
        """
        Simplified interface that returns a string result.
        
        Args:
            df: DataFrame to query
            question: Natural language question
            
        Returns:
            Formatted result string
        """
        response = self.query(df, question)
        
        if response['success']:
            mode = "🤖 Mock Mode" if response['method'] == 'mock' else "🌐 LLM Mode"
            output = f"{mode}\n\nQuestion: {question}\n\n"
            output += f"Generated Code:\n{response['code']}\n\n"
            output += f"Result:\n{response['result_str']}"
            return output
        else:
            return f"❌ Error: {response['error']}"


if __name__ == "__main__":
    # Demo usage
    from src.data.generator import generate_pos_data
    
    print("Generating synthetic data...")
    transactions, _ = generate_pos_data(num_days=30)
    
    print("\nInitializing query engine (mock mode for demo)...")
    engine = LLMQueryEngine(use_mock=True)
    
    # Test queries
    test_questions = [
        "What were the top 5 products by revenue?",
        "Show me sales by hour of day",
        "Which staff member had the most transactions?",
        "What was the average transaction value on weekends?",
        "How many refunds did we have?",
        "Show me sales by category",
    ]
    
    print("\n" + "="*80)
    for question in test_questions:
        print(f"\n{engine.ask(transactions, question)}")
        print("\n" + "="*80)
