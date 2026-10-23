import { copyFile, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const app = fileURLToPath(new URL('../', import.meta.url));
const atlasRoot = path.resolve(app, '../atlas/public/data');
const target = path.join(app, 'public/prototype');
try {
  const canonical = await realpath(atlasRoot);
  const catalog = JSON.parse(await readFile(path.join(canonical, 'catalog.json'), 'utf8'));
  if (catalog.schema !== 'malecns_lab.atlas.v1' || typeof catalog.atlas_url !== 'string') throw new Error('图谱目录 schema 不兼容');
  const relative = catalog.atlas_url;
  if (!/^[a-zA-Z0-9_./-]+$/.test(relative) || relative.split('/').some(part => !part || part === '.' || part === '..')) throw new Error('图谱路径无效');
  const source = await realpath(path.join(canonical, relative));
  if (!source.startsWith(`${canonical}${path.sep}`)) throw new Error('图谱越出资产目录');
  await mkdir(path.join(target, 'data'), { recursive: true });
  await copyFile(source, path.join(target, 'data/atlas.json'));
  // Preserve the source graph identity and provenance; diagnostic activity is never loaded.
  await writeFile(path.join(target, 'data/atlas-source.json'), `${JSON.stringify({ schema: catalog.schema, graph: catalog.graph, provenance: catalog.provenance })}\n`);
  await writeFile(path.join(target, 'manifest.json'), `${JSON.stringify({ schema: 'malecns_lab.desktop.assets.v1', assets: ['data/atlas.json', 'data/atlas-source.json'] }, null, 2)}\n`);
  process.stdout.write('已准备真实胞体坐标和图谱来源，未加载旧诊断活动。\n');
} catch (error) {
  process.stderr.write(`无法准备图谱：${error.message}\n请恢复现有 atlas 导出；不会生成替代数据。\n`);
  process.exitCode = 1;
}
