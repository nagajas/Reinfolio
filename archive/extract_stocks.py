# Libraries: 
# Selenium for web scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
# Utility libraries for timing jobs
from time import sleep, time
from tqdm import tqdm
import os
    
def download_dataset(driver, link)-> None:
    """Click button labelled "MAX" to get all historical data. Then click the download button.

    Args:
        driver (selenium.webdriver): Driver element.
        link (url): URL for the stock's download page.
    """
    driver.get(link)
    # Opening the dropdown menu via svg
    svg_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "svg[data-icon='CoreArrowDown']")))
    svg_element.click()
    # Seletin max option to get download link
    button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dropdown-menu"]/div/ul[2]/li[4]/button')))
    button.click()
    dwld_link = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a')))
    driver.get(dwld_link.get_attribute('href'))
    # Wait 3 seconds for download to start
    sleep(3)
    
    
def main():
    # Setting screen parameters (optinonal)
    width = 1920
    height = 1080
    BASE = "https://www.nasdaq.com"

    options = webdriver.ChromeOptions()
    # Redirect downloads to particular directory in case of link
    options.add_experimental_option("prefs", {
    "profile.default_content_settings.popups": 0,
    "download.prompt_for_download": False,
    "download.default_directory": r"~/Desktop/rl_trader_AI/data/", # This doesn't work with buttons, set download location manually on browser
    })

    driver = webdriver.Chrome(options=options)
    driver.set_window_size(width , height)
    driver.get("https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index")    

    stocks = driver.find_elements(By.XPATH,'/html/body/div[2]/div/main/div[2]/article/div[2]/div/div[3]/div[3]/div[2]/table/tbody/tr/th[1]/a')
    stock_tickers=[]
    for stock in stocks:
        stock_tickers.append(stock.text)
        
    for ticker in tqdm(stock_tickers):
        link = "https://finance.yahoo.com/quote/"+ticker+"/history"
        download_dataset(link)
        
    driver.quit()

if __name__ == "__main__":
    main()