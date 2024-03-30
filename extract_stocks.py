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
    
def download_dataset(driver, link):
    # 
    """Click button labelled "MAX" to get 10 years of data. Then click the download button.

    Args:
        driver (selenium.webdriver): Driver element.
        link (url): URL for the stock's download page.
    """
    driver.get(link+"/historical")
    button = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH,'/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/div/div/div/button[6]')))
    button.click()
    driver.find_element(By.XPATH,'/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/button').click()
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
    "download.default_directory": r"~/Desktop/StockData", # This only works with downloadable links and doesn't work with buttons, set download location manually on browser.
    })

    driver = webdriver.Chrome(options=options)
    driver.set_window_size(width , height)
    driver.get(BASE+"/market-activity/quotes/nasdaq-ndx-index")

    stocks = driver.find_elements(By.XPATH,'/html/body/div[2]/div/main/div[2]/article/div[2]/div/div[3]/div[3]/div[2]/table/tbody/tr/th[1]/a')
    stock_links = dict()
    for stock in stocks:
        #print(stock.text)
        stock_links[stock.text] = stock.get_attribute('href')
        
    for link in tqdm(list(stock_links.values())):
        download_dataset(driver, link)
        
    driver.quit()
    
    # Renaming the files to their respective ticker symbols.
    directory = 'DATA_10y/'
    files = os.listdir(directory)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    
    for file, stock in zip(files[1:], stock_links.keys()):
        if file != ".DS_Store":
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, stock + ".csv")
            os.rename(old_path, new_path)

if __name__ == "__main__":
    main()