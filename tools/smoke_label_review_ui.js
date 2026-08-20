const { chromium } = require('playwright');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:5058';
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('dialog', dialog => dialog.accept());

  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.getElementById('itemTitle').textContent !== '未选择');
  assert(await page.locator('.queue-item').count() === 6, 'review queue did not load');

  await page.click('#redrawButton');
  assert(await page.locator('#editActions').evaluate(node => node.classList.contains('visible')), 'edit panel not visible');
  await page.screenshot({ path: 'reports/label_review_editor_v2.png', fullPage: true });
  const originalPoint = await page.evaluate(() => app.draftContours[0][0].slice());
  const vertexScreen = await page.evaluate(() => {
    const point = app.draftContours[0][0];
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + app.offsetX + point[0] * app.item.width * app.scale,
      y: rect.top + app.offsetY + point[1] * app.item.height * app.scale
    };
  });
  await page.mouse.move(vertexScreen.x, vertexScreen.y);
  await page.mouse.down();
  await page.mouse.move(vertexScreen.x + 14, vertexScreen.y + 9, { steps: 3 });
  await page.mouse.up();
  const movedPoint = await page.evaluate(() => app.draftContours[0][0].slice());
  assert(movedPoint[0] !== originalPoint[0] || movedPoint[1] !== originalPoint[1], 'vertex drag did not change coordinates');

  const beforeInsert = await page.evaluate(() => app.draftContours[0].length);
  await page.click('#insertVertexTool');
  const edgeScreen = await page.evaluate(() => {
    const a = app.draftContours[0][0], b = app.draftContours[0][1];
    const point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + app.offsetX + point[0] * app.item.width * app.scale,
      y: rect.top + app.offsetY + point[1] * app.item.height * app.scale
    };
  });
  await page.mouse.click(edgeScreen.x, edgeScreen.y);
  assert(await page.evaluate(() => app.draftContours[0].length) === beforeInsert + 1, 'vertex insert failed');
  await page.click('#deleteVertexTool');
  await page.mouse.click(edgeScreen.x, edgeScreen.y);
  assert(await page.evaluate(() => app.draftContours[0].length) === beforeInsert, 'vertex delete failed');

  await page.click('#cancelRedraw');
  await page.click('#addButton');
  await page.click('#freehandToggle');
  const box = await page.locator('#canvas').boundingBox();
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.45);
  await page.mouse.down();
  for (const [dx, dy] of [[60, 0], [80, 45], [35, 80], [-20, 35], [-45, -20]]) {
    await page.mouse.move(box.x + box.width * 0.45 + dx, box.y + box.height * 0.45 + dy, { steps: 3 });
  }
  await page.mouse.up();
  assert(await page.evaluate(() => app.draftContours.length) === 1, 'freehand contour was not completed');
  await page.click('#undoPoint');
  assert(await page.evaluate(() => app.draftContours.length) === 0, 'local undo failed');
  await page.click('#redoPoint');
  assert(await page.evaluate(() => app.draftContours.length) === 1, 'local redo failed');

  await page.click('#saveDraftButton');
  await page.waitForFunction(() => document.getElementById('draftStatus').textContent.includes('已保存'));
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.getElementById('editActions').classList.contains('visible'));
  assert(await page.evaluate(() => app.editMode === 'add' && app.draftContours.length === 1), 'saved draft was not restored');
  await page.click('#cancelRedraw');
  await page.waitForFunction(() => !document.getElementById('editActions').classList.contains('visible'));

  const status = await page.evaluate(async () => (await fetch('/api/review/items')).json());
  assert(status.drafts === 0, 'smoke-test draft cleanup failed');
  assert(status.counts.pending === 6, 'smoke test changed review decisions');
  assert(errors.length === 0, `browser errors: ${errors.join(' | ')}`);
  console.log(JSON.stringify({ ok: true, queue: status.total, pending: status.counts.pending, drafts: status.drafts }));
  await browser.close();
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
