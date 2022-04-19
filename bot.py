from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import time 

DEFAULT_TIME_WAIT = 1
BASE_PATH = "/Users/enrico/Documents/Faculdade/TCC Local/Projeto/data"

def clear_reporters_list(browser, wait):
	x_button_selector = "#s2id_reporters > ul > li.select2-search-choice > a"
	wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, x_button_selector)))
	x_button = browser.find_element_by_css_selector(x_button_selector)
	x_button.click()
	time.sleep(DEFAULT_TIME_WAIT)

def x_button_and_find_list(browser, wait, combobox):
	clear_reporters_list(browser, wait)

	combobox.click()

	list_selector = "#select2-drop"
	list_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, list_selector)))
	return list_el

def select_all_partners(broswer, wait):
	x_button_selector = "#s2id_partners > ul > li.select2-search-choice > a"
	x_button = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, x_button_selector)))
	x_button.click()
	time.sleep(DEFAULT_TIME_WAIT)

	cb_selector = "#s2id_partners > ul"
	combobox = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, cb_selector)))

	combobox.click()

	list_selector = "#select2-drop"
	list_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, list_selector)))
	periods = list_el.find_elements_by_tag_name("div")
	periods[0].click()


def select_all_periods(browser, wait):
	monthly_selector = "#freq-m-lbl"
	monthly_rd = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, monthly_selector)))
	monthly_rd.click()

	time.sleep(DEFAULT_TIME_WAIT)

	x_button_selector = "#s2id_periods > ul > li.select2-search-choice > a"
	x_button = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, x_button_selector)))
	x_button.click()
	time.sleep(DEFAULT_TIME_WAIT)

	cb_selector = "#s2id_periods > ul"
	combobox = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, cb_selector)))

	combobox.click()

	list_selector = "#select2-drop"
	list_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, list_selector)))
	periods = list_el.find_elements_by_tag_name("div")
	periods[0].click()


def download(broswer, wait, country):
	country_path = f"{BASE_PATH}/{country}.csv"
	download_path = f"{BASE_PATH}/comtrade.csv"

	# don't download if country was already downloaded
	if os.path.isfile(country_path):
		return

	selector = "#download-csv-top"
	download_button = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
	download_button.click()

	# wait until file is downloaded
	i = 0
	while not os.path.isfile(download_path) and i < 180:
		i += 1
		time.sleep(DEFAULT_TIME_WAIT)

	if i == 180:
		raise Exception("File download timeout exceeded (180)")
	
	os.rename(download_path, country_path)

def get_last_country():
	files = os.listdir(BASE_PATH)
	if len(files) == 0:
		return None
	return sorted(files)[-1].replace('.csv', '')


def main():
	chrome_options = Options()
	prefs = {"download.default_directory": BASE_PATH}
	chrome_options.add_experimental_option("prefs", prefs)
	# chrome_options.add_argument("--headless")
	chrome_options.add_argument("--disable-dev-shm-usage")
	# chrome_options.add_argument("--no-sandbox")

	url = "https://comtrade.un.org/data/"

	if "GOOGLE_CHROME_BIN" in os.environ:
		chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
		browser= webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), options=chrome_options)
	else:
		browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

	browser.get(url)
	wait = WebDriverWait(browser, 10)

	select_all_periods(browser, wait)
	# select_all_partners(browser, wait)
	clear_reporters_list(browser, wait)

	cb_selector = "#s2id_reporters > ul"
	combobox = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, cb_selector)))

	combobox.click()

	list_selector = "#select2-drop"
	list_el = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, list_selector)))

	countries = list_el.find_elements_by_tag_name("div")
	last_country = get_last_country()
	found_country = False

	for i in range(len(countries)):
		# In case there is already countries in the folder
		# verify if current country is that last country, or skip
		if last_country != None:
			html = countries[i].get_attribute("innerHTML")
			if html.find(last_country) < 0:
				print("Skipping country")
				continue
			else:
				print("Found last country")
				last_country = None
				found_country = True
		if i > 0 and not found_country:
			list_el = x_button_and_find_list(browser, wait, combobox)
		country = list_el.find_elements_by_tag_name("div")[i]
		country.click()
		if found_country:
			found_country = False
		if i > 0:
			country_name = combobox.find_elements_by_tag_name("div")[0].get_attribute("innerHTML")
			print(f"Downloading {country_name}")
			download(browser, wait, country_name)


main()
# print(get_last_country())

	