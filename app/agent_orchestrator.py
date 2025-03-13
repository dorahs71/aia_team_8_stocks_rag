from typing import Protocol, Set, List, Iterator, Union
from dataclasses import dataclass
from expert_video_analysis import generate_answer
from long_short_analysis import analyze_stock
from basic_stock_analysis import process_stock_query
from institutional_trading_analysis import response_data
from morris_chang_analysis import response_morris_data

class Response:
    def __init__(self, type: str, content: Union[str, Iterator[str]]):
        self.type = type
        self.content = content

class Agent(Protocol):
    """Agent 介面"""
    @property
    def keywords(self) -> Set[str]:
        """代理關鍵字"""
        ...

    def process(self, query: str) -> Response:
        """處理查詢"""
        ...

@dataclass
class LongShortAnalysisAgent:
    """股票分析代理"""
    _keywords: Set[str] = frozenset({'多空走勢', '法人資訊','交易資訊'})

    @property
    def keywords(self) -> Set[str]:
        return self._keywords

    def process(self, query: str) -> Iterator[str]:
        print('我是 LongShortAnalysis')
        return analyze_stock(query)

@dataclass
class ExpertVideoAnalysisAgent:
    """專家分析代理"""
    _keywords: Set[str] = frozenset({'專家', '預測', '台股', '美股'})

    @property
    def keywords(self) -> Set[str]:
        return self._keywords

    def process(self, query: str) -> Iterator[str]:
        print('我是 ExpertVideoAnalysis')
        return generate_answer(query)
@dataclass 
class BasicStockAnalysisAgent:
    """專家分析代理"""
    _keywords: Set[str] = frozenset({'類股'})

    @property
    def keywords(self) -> str:
        return self._keywords

    def process(self, query: str) -> str:
        print('我是 BasicStockAnalysis')
        return process_stock_query(query)
    
@dataclass 
class InstitutionalTradingAnalysisAgent:
    """專家分析代理"""
    _keywords: Set[str] = frozenset({'法人買賣', '融資餘額'})

    @property
    def keywords(self) -> str:
        return self._keywords

    def process(self, query: str) -> Iterator[str]:
        print('我是 InstitutionalTradingAnalysisAgent')
        return response_data(query)

@dataclass 
class MorrisChangAnalysisAgent:
    """專家分析代理"""
    _keywords: Set[str] = frozenset({'張忠謀'})

    @property
    def keywords(self) -> str:
        return self._keywords

    def process(self, query: str) -> str:
        print('我是 MorrisChangAnalysisAgent')
        return response_morris_data(query)


class AgentOrchestrator:
    """代理編排器"""
    def __init__(self):
        self.agents: List[Agent] = [
            LongShortAnalysisAgent(),
            ExpertVideoAnalysisAgent(),
            BasicStockAnalysisAgent(),
            InstitutionalTradingAnalysisAgent(),
            MorrisChangAnalysisAgent()
        ]
        self.default_agent = BasicStockAnalysisAgent()

    def get_agent(self, query: str) -> Agent:
        """根據查詢選擇合適的代理"""
        for agent in self.agents:
            if any(keyword in query for keyword in agent.keywords):
                return agent
        return self.default_agent

    def process_query(self, query: str) -> Response:
        """處理查詢並返回結果"""
        # try:
        #     agent = self.get_agent(query)
        #     yield from agent.process(query)  # 使用 yield from 直接傳遞生成器
        # except Exception as e:
        #     yield f"抱歉，處理您的請求時發生錯誤：{str(e)}" 
        
        try:
            agent = self.get_agent(query)
            response = agent.process(query)
            if isinstance(response, str):  # 檢查是否為純字串
                return Response("string", agent.process(query))
            else:
                return Response("generator", agent.process(query))
        except Exception as e:
            return Response("string", f"抱歉，處理您的請求時發生錯誤：{str(e)}")