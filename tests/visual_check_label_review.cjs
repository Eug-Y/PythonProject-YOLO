const path = require('path');

const playwrightPath = process.argv[2];
const executablePath = process.argv[3];
const baseUrl = process.argv[4] || 'http://127.0.0.1:5055';
const outputDir = process.argv[5] || 'reports';
const { chromium } = require(playwrightPath);

async function checkCanvas(page) {
  await page.waitForFunction(() => {
    const canvas = document.getElementById('canvas');
    const empty = document.getElementById('emptyState');
    return canvas && canvas.width > 100 && canvas.height > 100 && empty.style.display === 'none';
  });
  const pixels = await page.locator('#canvas').evaluate((canvas) => {
    const context = canvas.getContext('2d');
    const width = Math.min(canvas.width, 500);
    const height = Math.min(canvas.height, 500);
    const data = context.getImageData(0, 0, width, height).data;
    let opaque = 0;
    let varied = 0;
    for (let index = 0; index < data.length; index += 16) {
      if (data[index + 3] > 0) opaque += 1;
      if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) varied += 1;
    }
    return { opaque, varied };
  });
  if (pixels.opaque < 1000 || pixels.varied < 1000) {
    throw new Error(`Canvas appears blank: ${JSON.stringify(pixels)}`);
  }
  return pixels;
}

async function checkImportantBounds(page) {
  return page.evaluate(() => {
    const selectors = ['header h1', '#progressText', '#queueList', '#canvasWrap', '.inspector', '#normalActions'];
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    return selectors.map((selector) => {
      const element = document.querySelector(selector);
      const rect = element.getBoundingClientRect();
      return {
        selector,
        visible: rect.width > 0 && rect.height > 0,
        left: Math.round(rect.left), top: Math.round(rect.top),
        right: Math.round(rect.right), bottom: Math.round(rect.bottom),
        viewport,
      };
    });
  });
}

(async () => {
  const browser = await chromium.launch({ executablePath, headless: true });
  const errors = [];
  const results = {};
  try {
    for (const profile of [
      { name: 'desktop', viewport: { width: 1440, height: 900 } },
      { name: 'mobile', viewport: { width: 390, height: 844 } },
    ]) {
      const context = await browser.newContext({ viewport: profile.viewport, deviceScaleFactor: 1 });
      const page = await context.newPage();
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`${profile.name}: ${message.text()}`);
      });
      page.on('pageerror', (error) => errors.push(`${profile.name}: ${error.message}`));
      await page.goto(baseUrl, { waitUntil: 'networkidle' });
      const canvas = await checkCanvas(page);
      const bounds = await checkImportantBounds(page);
      await page.screenshot({ path: path.join(outputDir, `label_review_${profile.name}.png`), fullPage: true });

      if (profile.name === 'desktop') {
        await page.locator('#redrawButton').click();
        const canvasBox = await page.locator('#canvas').boundingBox();
        for (const [x, y] of [[0.42, 0.42], [0.52, 0.42], [0.48, 0.53]]) {
          await page.mouse.click(canvasBox.x + canvasBox.width * x, canvasBox.y + canvasBox.height * y);
        }
        await page.locator('#finishContour').click();
        for (const [x, y] of [[0.57, 0.47], [0.66, 0.47], [0.62, 0.58]]) {
          await page.mouse.click(canvasBox.x + canvasBox.width * x, canvasBox.y + canvasBox.height * y);
        }
        const count = await page.locator('#editCount').textContent();
        const saveText = await page.locator('#saveRedraw').textContent();
        const saveDisabled = await page.locator('#saveRedraw').isDisabled();
        if (!count.includes('已完成 1') || !count.includes('当前 3') || !saveText.includes('2') || saveDisabled) {
          throw new Error(`Multi-contour redraw failed: count=${count}, save=${saveText}, disabled=${saveDisabled}`);
        }
        await page.locator('#cancelRedraw').click();
      }
      results[profile.name] = { canvas, bounds };
      await context.close();
    }
    if (errors.length) throw new Error(`Browser errors: ${errors.join(' | ')}`);
    process.stdout.write(JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
