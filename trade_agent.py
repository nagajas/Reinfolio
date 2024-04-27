from agent import TradeAgent

def main(*args):
    """
        Module to predict stock trends.
    """
    if len(args) == 0 or len(args) > 5:
        print("Usage: python trade_agent.py [stock] [lookback_value] [features] [batch_size] [epochs]")
        return
    
    stock = args[0]
    lookback = 30
    features = 8
    batch_size = 32
    epochs = 30
    
    if len(args) > 1:
        lookback = args[1]
    if len(args) > 2:
        features = args[2]
    if len(args) > 3:
        batch_size = args[3]
    if len(args) > 4:
        epochs = args[4]
        
    agent = TradeAgent(stock, lookback, features)
    agent.train(batch_size, epochs)
    agent.save_model()
    agent.evaluate()

if __name__ == '__main__':
    main()
