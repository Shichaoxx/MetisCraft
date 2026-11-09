/* Browser acceptance for the built community documentation. Requires Playwright. */
'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const args = Object.fromEntries(process.argv.slice(2).reduce((out, item, i, all) => {
  if (item.startsWith('--')) out.push([item.slice(2), all[i + 1]]);
  return out;
}, []));
if (!args.module || !args.browser || !args.site || !args.output) {
  throw new Error('Required: --module PLAYWRIGHT --browser CHROMIUM --site SITE --output DIRECTORY');
}
const { chromium } = require(path.resolve(args.module));
const site = path.resolve(args.site), output = path.resolve(args.output);
fs.mkdirSync(output, { recursive: true });
const types = {'.html':'text/html', '.js':'application/javascript', '.css':'text/css', '.svg':'image/svg+xml', '.png':'image/png', '.woff2':'font/woff2', '.woff':'font/woff', '.ttf':'font/ttf', '.json':'application/json', '.pdf':'application/pdf', '.tex':'text/plain'};
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (!url.pathname.startsWith('/MetisCraft/')) { res.writeHead(404).end(); return; }
  let target = path.resolve(site, decodeURIComponent(url.pathname.slice('/MetisCraft/'.length)));
  if (!target.startsWith(site + path.sep) && target !== site) {res.writeHead(403).end();return;}
  if (fs.existsSync(target) && fs.statSync(target).isDirectory()) target = path.join(target,'index.html');
  if (!fs.existsSync(target)) {res.writeHead(404).end();return;}
  res.writeHead(200, {'Content-Type': types[path.extname(target)] || 'application/octet-stream'});
  fs.createReadStream(target).pipe(res);
});
const report = { pages: [], layouts: [], interactions: [], failures: [], externalRequests: [] };
function ensure(ok, detail) { if (!ok) report.failures.push(detail); }
(async () => {
  let browser;
  try {
    await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
    const origin = 'http://127.0.0.1:' + server.address().port;
    browser = await chromium.launch({executablePath:path.resolve(args.browser),args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu']});
    const context = await browser.newContext({viewport:{width:1440,height:1080},permissions:['clipboard-read','clipboard-write']});
    await context.route('**/*', route => {
      const url = route.request().url();
      if (url.startsWith(origin) || url.startsWith('data:') || url.startsWith('blob:')) return route.continue();
      report.externalRequests.push(url); return route.abort();
    });
    const page = await context.newPage();
    page.on('pageerror', e => report.failures.push('pageerror: '+e.message));
    const mapText = fs.readFileSync(path.join(site,'assets/language-map.js'),'utf8');
    const mapping = JSON.parse(mapText.slice(mapText.indexOf('=')+1).trim().replace(/;$/,''));
    async function go(route) {
      const response=await page.goto(origin+route,{waitUntil:'load'});
      ensure(response.status()===200,route+' HTTP '+response.status());
      await page.waitForFunction(()=>window.renderMetisMath && document.querySelector('.md-content'));
      await page.evaluate(()=>window.renderMetisMath());
      await page.evaluate(()=>document.fonts.ready);
    }
    async function theme(mode) {
      const input=page.locator('input[data-md-color-scheme="'+mode+'"]');
      if (await input.count()) {
        const current=await page.locator("body").getAttribute("data-md-color-scheme");
        if(current!==mode) await page.locator('label[for="'+await input.getAttribute("id")+'"]').click();
      }
      else await page.evaluate(m=>document.body.setAttribute('data-md-color-scheme',m),mode);
      await page.waitForFunction(m=>document.body.getAttribute('data-md-color-scheme')===m,mode);
      await page.waitForFunction(m=>document.querySelector('link[rel="icon"]').href.includes(m==='slate'?'mark-dark.svg':'mark-light.svg'),mode);
    }
    for (const route of Object.keys(mapping)) {
      if(args.mode==='themes') continue;
      if(args.mode==='interactions' && !route.includes('/recipes/tev1/')) continue;
      await go(route);
      const facts=await page.evaluate(()=>({
        math:document.querySelectorAll('.arithmatex').length,
        failedMath:[...document.querySelectorAll('.arithmatex')].filter(n=>n.dataset.mathRendered!=='true').length,
        missingImages:[...document.querySelectorAll('.md-content img')].filter(n=>!n.complete||!n.naturalWidth).map(n=>n.src),
        language:[...document.querySelectorAll('a[hreflang]')].map(n=>({lang:n.hreflang,path:new URL(n.href).pathname})),
        suffix:/（中文）|\(Chinese\)|\(中文\)/.test(document.querySelector('.md-sidebar--primary').textContent)
      }));
      ensure(!facts.failedMath,route+' failed math '+facts.failedMath);
      ensure(!facts.missingImages.length,route+' missing images '+facts.missingImages.join(','));
      ensure(!facts.suffix,route+' language suffix in navigation');
      for (const [lang, partner] of Object.entries(mapping[route])) ensure(facts.language.some(n=>n.lang===lang&&n.path===partner),route+' missing static language partner '+lang);
      report.pages.push({route,...facts});
    }
    const slugs=['','cookbook/','tensor-analysis/','tensor-analysis/answer-token/','tensor-analysis/scalar-heads/','tensor-analysis/training-calibration/','tensor-analysis/readout-comparison/','tensor-analysis/notation/','tensor-analysis/walkthrough/'];
    if(!['interactions','themes'].includes(args.mode)) {
    for (const lang of ['en','zh']) for (const width of [1440,390]) for (const mode of ['default','slate']) {
      await page.setViewportSize({width,height:1080});
      for (const slug of slugs) {
        const route='/MetisCraft/'+lang+'/'+slug;
        await go(route);await theme(mode);
        const facts=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,images:[...document.querySelectorAll('.md-content img')].every(n=>n.complete&&n.naturalWidth),mathErrors:document.querySelectorAll('.math-render-error').length,logo:document.querySelector('.md-logo img')?.src,favicon:document.querySelector('link[rel="icon"]').href}));
        ensure(facts.scroll<=width+1,route+' horizontal overflow '+width+' '+mode+': '+facts.scroll);
        ensure(facts.images&&!facts.mathErrors,route+' layout image/math failure');
        const file=lang+'-'+(slug.replaceAll('/','-')||'home')+'-'+width+'-'+mode+'.png';
        await page.screenshot({path:path.join(output,file)});
        report.layouts.push({route,width,mode,file,...facts});
      }
    }
    }
    // A stale manual preference must never override the page's actual theme.
    await go('/MetisCraft/en/cookbook/');
    await page.evaluate(()=>localStorage.setItem('metiscraft.metis-outfit','gown'));
    for(const lang of ['en','zh']) for(const width of [1440,390]) {
      await page.setViewportSize({width,height:1080});
      for(const slug of ['','decision/','first-model/','own-task/']) {
        const route='/MetisCraft/'+lang+'/cookbook/'+slug;
        let lightGeometry;
        for(const mode of ['default','slate']) {
          await go(route);await theme(mode);
          const expected=mode==='slate'?'dark':'light';
          await page.waitForFunction(t=>[...document.querySelectorAll('[data-metis-theme]')].every(n=>n.dataset.metisTheme===t),expected);
          await page.evaluate(async()=>{for(const img of document.querySelectorAll('img[data-metis-portrait]'))await img.decode();});
          const facts=await page.evaluate(()=>({
            controls:document.querySelectorAll('[data-metis-outfit-choice],.metis-wardrobe-controls').length,
            scroll:document.documentElement.scrollWidth,
            guides:[...document.querySelectorAll('[data-metis-theme]')].map(n=>{
              const img=n.querySelector('img[data-metis-portrait]'),box=img?.getBoundingClientRect(),css=img&&getComputedStyle(img);
              return{theme:n.dataset.metisTheme,src:img?.src,alt:img?.alt,width:box?.width,height:box?.height,fit:css?.objectFit,transform:css?.transform,full:n.querySelector('a[data-metis-full]')?.href};
            })
          }));
          ensure(facts.controls===0,route+' independent outfit control remains');
          ensure(facts.scroll<=width+1,route+' guide overflows '+width+' '+mode);
          ensure(facts.guides.length===1,route+' expected one reading guide');
          const expectedSize=slug?(width<=600?100:120):160;
          for(const guide of facts.guides) {
            ensure(guide.src.endsWith(expected==='dark'?'metis-guide-night.png':'metis-guide-chibi.png'),route+' wrong themed portrait');
            ensure(guide.width===expectedSize&&guide.height===expectedSize,route+' unexpected portrait size '+guide.width+'x'+guide.height);
            ensure(guide.fit==='contain'&&guide.transform==='none',route+' portrait crop or scale');
            ensure(guide.full===guide.src&&Boolean(guide.alt),route+' original link or alt missing');
          }
          const geometry=facts.guides.map(g=>[g.width,g.height]);
          if(mode==='default')lightGeometry=geometry;
          else ensure(JSON.stringify(lightGeometry)===JSON.stringify(geometry),route+' day/night geometry differs');
          const file=lang+'-theme-guide-'+(slug.replaceAll('/','')||'index')+'-'+width+'-'+mode+'.png';
          await page.screenshot({path:path.join(output,file)});
          report.layouts.push({route,width,mode,file,...facts});
        }
      }
    }
    await page.reload({waitUntil:'load'});
    await page.waitForFunction(()=>document.querySelector('[data-metis-theme]')?.dataset.metisTheme==='dark');
    ensure((await page.locator('img[data-metis-portrait]').getAttribute('src')).endsWith('metis-guide-night.png'),'Theme-only portrait after reload');
    report.interactions.push('Theme-only portraits ignore old manual preference and preserve size across 32 layouts');
    await page.setViewportSize({width:1440,height:1080});
    await go('/MetisCraft/en/tensor-analysis/scalar-heads/');await theme('default');
    const figure=page.locator('.md-content img[data-figure-ready]').first();
    await figure.scrollIntoViewIfNeeded();await figure.press('Enter');
    ensure(await page.locator('#metis-figure-dialog').evaluate(n=>n.open),'Figure keyboard open');
    const figUrl=await page.locator('#metis-figure-dialog > a').getAttribute('href');
    ensure((await context.request.get(figUrl)).status()===200,'Figure original resource');
    await page.screenshot({path:path.join(output,'figure-zoom.png')});
    await page.keyboard.press('Escape');
    ensure(!(await page.locator('#metis-figure-dialog').evaluate(n=>n.open)),'Figure Escape close');
    report.interactions.push('Keyboard figure zoom, original SVG, Escape close');
    for (const suffix of ['.svg','.pdf','.tex']) {
      const href=await page.locator('.md-content a[href$="'+suffix+'"]').first().getAttribute('href');
      const response=await context.request.get(new URL(href,page.url()).href);
      ensure(response.status()===200,'Download '+suffix);report.interactions.push('Download '+suffix);
    }
    const copy=page.locator('.md-code__button[data-md-type="copy"]').first();
    await copy.click();
    const copiedTarget=await copy.getAttribute('data-clipboard-target');
    const expectedCode=await page.locator(copiedTarget).textContent();
    const clip=await page.evaluate(()=>navigator.clipboard.readText());ensure(clip.trim()===expectedCode.trim(),'Copy code clipboard');report.interactions.push('Copy code clipboard');
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.getByRole('button',{name:'Select language'}).hover();
    await page.locator('a[hreflang="zh"]').click();
    await page.waitForURL(url=>url.pathname==='/MetisCraft/zh/tensor-analysis/scalar-heads/');report.interactions.push('Same-page language switch');
    for(const anchor of ['记号','①-lm-head-答案-token','③-逐候选标量头','未完成']) {
      await page.goto(origin+'/MetisCraft/zh/readouts/#'+encodeURIComponent(anchor),{waitUntil:'load'});
      await page.waitForURL(url=>url.pathname==='/MetisCraft/zh/tensor-analysis/readout-comparison/');
      ensure(decodeURIComponent(new URL(page.url()).hash.slice(1))===anchor,'Legacy fragment retained '+anchor);
      ensure(await page.evaluate(id=>Boolean(document.getElementById(id)),anchor),'Legacy anchor exists '+anchor);
      report.interactions.push('Legacy anchor '+anchor);
    }
    await go('/MetisCraft/en/tensor-analysis/answer-token/');
    const search=page.locator('.md-search__button');
    if(await search.count()) await search.click(); else await page.keyboard.press('Control+k');
    await page.waitForTimeout(250);
    const input=page.locator('input[type="search"], .md-search input, dialog input, [role="dialog"] input').first();
    if(await input.count()) {await input.fill('scalar');await page.waitForTimeout(400);await page.waitForTimeout(750);
      const searchFacts=await page.evaluate(()=>{
        function dialogs(root) {let found=[...root.querySelectorAll('dialog,[role=dialog],.md-search')];for(const node of root.querySelectorAll('*'))if(node.shadowRoot)found.push(...dialogs(node.shadowRoot));return found;}
        return {dialogs:dialogs(document).map(n=>({tag:n.tagName,text:n.textContent.slice(0,6000)}))};
      });
      fs.writeFileSync(path.join(output,'search-inspection.json'),JSON.stringify(searchFacts,null,2));
      ensure(searchFacts.dialogs.some(n=>n.text.toLowerCase().includes('scalar')),'Search scalar results');report.interactions.push('Search scalar results');await page.keyboard.press('Escape');}
    else ensure(false,'Search input missing');
    await page.setViewportSize({width:320,height:900});await go('/MetisCraft/zh/tensor-analysis/scalar-heads/');
    ensure(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'320px overflow');report.interactions.push('320px narrow layout');
    const retina=await browser.newContext({viewport:{width:1440,height:1080},deviceScaleFactor:2});
    await retina.route('**/*',route=>route.request().url().startsWith(origin)?route.continue():route.abort());
    const rp=await retina.newPage();
    await rp.goto(origin+'/MetisCraft/zh/tensor-analysis/answer-token/');
    await rp.waitForFunction(()=>window.renderMetisMath && document.querySelector('.md-content img[data-figure-ready]'));
    const ri=rp.locator('.md-content img[data-figure-ready]').first();await ri.scrollIntoViewIfNeeded();await ri.click();
    ensure(await rp.locator('#metis-figure-dialog').evaluate(n=>n.open),'2x vector enlargement');
    await rp.screenshot({path:path.join(output,'vector-retina-2x.png')});
    report.interactions.push('2x display vector enlargement');await retina.close();
    await context.close();
    report.status=report.failures.length?'fail':'pass';
  } catch(error) {report.status='fail';report.failures.push(error.stack||String(error));}
  finally {if(browser) await browser.close();await new Promise(resolve=>server.close(resolve));fs.writeFileSync(path.join(output,'validation.json'),JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify({status:report.status,pages:report.pages.length,layouts:report.layouts.length,interactions:report.interactions.length,failures:report.failures},null,2));
  if(report.status!=='pass')process.exitCode=1;
})();
