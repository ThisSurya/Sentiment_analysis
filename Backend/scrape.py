# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By
# from selenium.common import TimeoutException
# from time import sleep

# option = webdriver.ChromeOptions()
# option.add_argument('--headless=new')



# def get_comment():
#     # chromeservice = ChromeService(ChromeDriverManager.install())
#     driver = webdriver.Chrome(
#         # service=chromeservice,  
#         options=option
#     )
#     print("initial complete start get url...")
#     driver.set_page_load_timeout(10)
#     url = 'https://www.youtube.com/watch?v=kuDuJWvho7Q'

#     print("getting url...")
#     driver.get(url=url)

#     # try:
#     #     consent_overlay = WebDriverWait(driver, 15).until(
#     #         EC.visibility_of_element_located((By.CSS_SELECTOR, 'h1.ytd-watch-metadata'))
#     #     )

#     #     consent_button = consent_overlay.find_elements(By.CSS_SELECTOR, '.eom-buttons button.yt-spec-button-shape-next')
#     #     if consent_button > 1 :
#     #         accept_all = consent_button[1]
#     #         accept_all.click()
#     # except TimeoutException:
#     #     print("Cookie modal is missing!!")

#     # video = {}

#     # comment_scection = driver.find_element(By.CSS_SELECTOR, 'ytd-comments.style-scope')
#     # comment_1 = comment_scection.find_elements()

#     for x in range(25):
#         driver.execute_script("window.scrollBy(0,500)","")
#         sleep(2)
#     sleep(10)

#     comment_section = driver.find_elements(By.XPATH, """//*[@id="content-text"]/span""")
#     comment_list = []
#     for i in comment_section:
#         comment_list.append(i.text)


#     # after scraping completed...
#     driver.quit()

#     return comment_list




