import fs from 'fs';
const src = fs.readFileSync(new URL('../web/ripplechunks_preview.js', import.meta.url), 'utf8');
class El {
  constructor(tag) { this.tag = tag; this.style = {}; this.children = []; this.listeners = {}; this.textContent = ''; this.naturalWidth = 0; this.naturalHeight = 0; }
  append(...c) { this.children.push(...c); } appendChild(c) { this.children.push(c); }
  addEventListener(t, f) { this.listeners[t] = f; }
  fire(t) { this.listeners[t] && this.listeners[t](); }
}
const store = {}, handlers = {};
globalThis.localStorage = { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
globalThis.document = { createElement: t => new El(t), body: new El('body') };
const winListeners = {};
globalThis.window = { innerWidth: 1920, innerHeight: 1080, addEventListener: (n, f) => { winListeners[n] = f; },
  comfyAPI: { api: { api: { apiURL: p => '/api' + p, addEventListener: (n, f) => { handlers[n] = f; } } } } };
await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
await new Promise(r => setTimeout(r, 20));
const find = (el, pred) => pred(el) ? el : el.children.map(c => find(c, pred)).find(Boolean);
const num = v => parseInt(v, 10);
handlers['ripplechunks.preview']({ detail: { filename: 'a.webp', type: 'temp', label: 'chunk 1 of 2 done' } });
const box = document.body.children[0];
const img = find(box, e => e.tag === 'img');
// portrait preview 270x480 arrives
img.naturalWidth = 270; img.naturalHeight = 480; img.fire('load');
const w = () => num(img.style.width);
if (num(box.style.width) !== w() + 16) throw new Error(`window ${box.style.width} is not picture ${w()} + padding`);
if (w() !== 360) throw new Error('default size wrong: ' + w());
const slider = find(box, e => e.tag === 'input');
for (const size of [160, 240, 400]) {
  slider.value = String(size); slider.fire('input');
  if (w() !== size || num(box.style.width) !== size + 16) throw new Error(`size ${size}: picture ${w()} window ${box.style.width}`);
}
// a huge size is limited so the picture fits the screen height (1080*0.88-80)/(480/270)
slider.value = '1000'; slider.fire('input');
const limit = Math.round((1080 * 0.88 - 80) / (480 / 270));
if (w() !== limit) throw new Error(`tall picture not limited: ${w()} vs ${limit}`);
// small screen: width never exceeds the screen
window.innerWidth = 300; winListeners['resize'](); if (num(box.style.width) > 300) throw new Error('wider than screen');
// the title cannot widen the window
const title = find(box, e => e.tag === 'span');
if (title.style.overflow !== 'hidden' || title.style.whiteSpace !== 'nowrap' || title.style.minWidth !== '0') throw new Error('title can push the window wider');
// next preview keeps the size and re-fits to the new picture
window.innerWidth = 1920; slider.value = '300'; slider.fire('input');
handlers['ripplechunks.preview']({ detail: { filename: 'b.webp', type: 'temp', label: 'chunk 2 of 2 done' } });
img.naturalWidth = 400; img.naturalHeight = 400; img.fire('load');
if (w() !== 300 || num(box.style.width) !== 316) throw new Error('did not re-fit: ' + w());
// position, settings memory, close and reopen
const where = find(box, e => e.tag === 'select');
if (box.style.right !== '12px') throw new Error('default position should be right');
where.value = 'center'; where.fire('change');
if (box.style.left !== '50%' || box.style.transform !== 'translateX(-50%)') throw new Error('center wrong');
where.value = 'left'; where.fire('change');
if (box.style.left !== '12px') throw new Error('left wrong');
const saved = JSON.parse(store['RippleChunks.preview.v1']);
if (saved.size !== 300 || saved.position !== 'left') throw new Error('settings not saved: ' + JSON.stringify(saved));
find(box, e => e.tag === 'button').fire('click');
if (box.style.display !== 'none') throw new Error('close broken');
handlers['ripplechunks.preview']({ detail: { filename: 'c.webp', type: 'temp' } });
if (box.style.display !== 'block') throw new Error('did not reopen');
const url = find(box, e => e.tag === 'img').src;
if (!/^\/api\/view\?filename=c\.webp&subfolder=&type=temp&rand=/.test(url)) throw new Error('bad url ' + url);
console.log('window width always = picture width + padding (resizes with the slider, fits the screen, title cannot widen it): OK');
