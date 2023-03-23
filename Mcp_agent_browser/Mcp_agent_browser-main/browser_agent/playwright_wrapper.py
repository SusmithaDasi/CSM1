from playwright.async_api import async_playwright

class BrowserControllerAsync:
    def __init__(self):
        self.browser = None
        self.page = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def visit(self, url):
        if not url.startswith("http"):
            url = "https://" + url
        await self.page.goto(url)
        return f"Visited {url}"

    async def click(self, selector):
        try:
            await self.page.click(selector)
            return f"Clicked on {selector}"
        except Exception as e:
            return f"Failed to click {selector}: {str(e)}"

    async def scroll(self, amount=1000):
        await self.page.evaluate(f"window.scrollBy(0, {amount})")
        return f"Scrolled {amount}px"

    async def screenshot(self, filename="screenshot.png"):
        await self.page.screenshot(path=filename)
        return filename

    async def extract_text(self, selector="body"):
        try:
            element = await self.page.query_selector(selector)
            if element:
                text = await element.inner_text()
                return text
            else:
                return "No element found with selector."
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    async def close(self):
        await self.browser.close()
