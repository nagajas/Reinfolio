from agent import TradeAgent

def main(*args):
    if len(args) == 0 or len(args) > 5:
        print("Usage: python trade_agent.py [stock] [lookback_value] [features] [batch_size] [epochs]")
        return
    
    stock = args[0]
    lookback = 30
    features = 8
    
    if len(args) > 1:
        lookback = args[1]
    if len(args) > 2:
        features = args[2]
        
    agent = TradeAgent(stock, lookback, features)
    agent.train(32, 30)
    agent.save_model()
    agent.evaluate()

if __name__ == '__main__':
    main()
