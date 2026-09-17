const path = require('path');

const playwrightPath = process.argv[2];
const baseUrl = process.argv[3] || 'http://127.0.0.1:5057';
const { chromium } = require(playwrightPath);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  try {
    await page.goto(baseUrl, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => typeof app !== 'undefined' && app.item && app.image);
    for (const id of ['toggleNumbers', 'editVertices', 'qualityCheck']) {
      assert(await page.locator(`#${id}`).isVisible(), `${id} is not visible`);
    }

    const report = await page.evaluate(async () => {
      const response = await fetch(`/api/item/${encodeURIComponent(app.item.image)}/quality`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({polygons: app.polygons})
      });
      return response.json();
    });
    assert(report.ok && report.polygon_count > 0, 'quality report did not inspect polygons');

    await page.evaluate(() => { app.selected = 0; updateUi(); draw(); });
    const before = await page.evaluate(() => app.polygons[0].points[0].slice());
    await page.click('#editVertices');
    const vertex = await page.evaluate(() => {
      const point = screenPoint(app.polygons[0].points[0]);
      const rect = canvas.getBoundingClientRect();
      return {x: rect.left + point[0], y: rect.top + point[1]};
    });
    await page.mouse.move(vertex.x, vertex.y);
    await page.mouse.down();
    await page.mouse.move(vertex.x + 7, vertex.y + 5, {steps: 3});
    await page.mouse.up();
    const after = await page.evaluate(() => app.polygons[0].points[0].slice());
    assert(before[0] !== after[0] || before[1] !== after[1], 'vertex drag did not update the contour');

    await page.waitForTimeout(800);
    const cleanup = await page.evaluate(async () => {
      clearTimeout(app.draftTimer);
      const response = await fetch(`/api/item/${encodeURIComponent(app.item.image)}/draft`, {method: 'DELETE'});
      app.dirty = false;
      app.draftSaved = true;
      return response.json();
    });
    assert(cleanup.ok, 'temporary smoke-test draft cleanup failed');
    assert(errors.length === 0, `browser errors: ${errors.join(' | ')}`);
    process.stdout.write(JSON.stringify({ok: true, polygonCount: report.polygon_count, vertexDrag: true}));
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
