from playwright.sync_api import sync_playwright
from datetime import datetime

print("Starting the Playwright script...")
print(f'script started at: {datetime.now()}')

#daily weather report bot
#chromium - > weather site -> extract the report -> screen shot -> final text file
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://weather.com/en-IN/weather/today/l/INXX0001:1:IN")
    page.wait_for_load_state("networkidle")  
    page.screenshot(path="weather_report.png")
    weather_report = page.inner_text("div.CurrentConditions--phraseValue--2xXSr")
    with open("weather_report.txt", "w") as f:
        f.write(weather_report)
    print(f'Weather report extracted at: {datetime.now()}')
    print(f'Weather report: {weather_report}')
    browser.close()         
print(f'script ended at: {datetime.now()}')

