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
  const structureSource = path.resolve(app, '../../data/brain_visualization_v1');
  const structure = JSON.parse(await readFile(path.join(structureSource, 'manifest.json'), 'utf8'));
  if (structure.schema !== 'malecns_lab.brain_display.v1' || structure.provenance.graph_sha256 !== catalog.graph.graph_sha256) throw new Error('三维展示资产与图谱连接组身份不一致');
  const names = ['manifest.json', ...Object.keys(structure.files)];
  await mkdir(path.join(target, 'brain'), { recursive: true });
  for (const name of names) {
    if (!/^[a-z0-9-]+\.(json|f32|u32|u16)$/.test(name)) throw new Error('展示资产文件名无效');
    await copyFile(path.join(structureSource, name), path.join(target, 'brain', name));
  }
  await writeFile(path.join(target, 'manifest.json'), `${JSON.stringify({ schema: 'malecns_lab.desktop.assets.v1', assets: ['data/atlas.json', 'data/atlas-source.json', ...names.map(name => `brain/${name}`)] }, null, 2)}\n`);
  process.stdout.write('已准备真实胞体、突触抽样、连接示意和历史电流片段。\n');
} catch (error) {
  process.stderr.write(`无法准备图谱：${error.message}\n请恢复现有 atlas 与 data/brain_visualization_v1 导出；不会生成替代数据。\n`);
  process.exitCode = 1;
}
